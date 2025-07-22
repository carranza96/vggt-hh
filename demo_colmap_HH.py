# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square, load_and_preprocess_images_no_resize
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
import matplotlib.pyplot as plt
# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### Image loading parameters #########
    parser.add_argument("--sort_images", action="store_true", default=False, 
                       help="Sort images by filename (default: unsorted for better diversity)")
    parser.add_argument("--use_known_intrinsics", action="store_true", default=False,
                       help="Use known calibrated intrinsics instead of VGGT predictions")
    parser.add_argument("--load_imgs_squared", action="store_true", default=True,
                       help="Load images with square padding and resizing (default: True). Use --no-load_imgs_squared for original resolution")
    parser.add_argument("--img_load_resolution", type=int, default=1024,
                       help="Target resolution for image loading when --load_imgs_squared is True")
    parser.add_argument("--vggt_resolution", type=int, default=518,
                       help="Resolution used by VGGT model (default: 518)")
    parser.add_argument("--undistort_images", action="store_true", default=False,
                       help="Undistort images using known camera intrinsics before processing")
    
    ######### Image selection parameters #########
    parser.add_argument("--max_images", type=int, default=None,
                       help="Maximum number of images to use (default: use all images)")
    parser.add_argument("--image_selection_method", type=str, default="first", 
                       choices=["first", "last", "uniform"],
                       help="Method for selecting images: 'first' (first N), 'last' (last N), 'uniform' (uniformly spaced)")
    
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518, maintain_aspect_ratio=False):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    if maintain_aspect_ratio:
        # Instead of forcing square, maintain aspect ratio but ensure larger dimension is target_resolution
        B, C, H, W = images.shape
        if H >= W:
            new_H = resolution
            new_W = int(W * (resolution / H))
        else:
            new_W = resolution
            new_H = int(H * (resolution / W))
        
        # Ensure dimensions are divisible by 14 (patch size)
        new_H = (new_H // 14) * 14
        new_W = (new_W // 14) * 14
        
        images = F.interpolate(images, size=(new_H, new_W), mode="bilinear", align_corners=False)
        print(f"Resized maintaining aspect ratio to: {images.shape}")
    else:
        # Standard square resizing
        images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
        print(f"Resized to square: {images.shape}")
    
    # Now move to GPU after resizing
    device = next(model.parameters()).device
    images = images.to(device)
    
    plt.figure()
    plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
    plt.savefig("sample_image_vggt.png")
    
    # Log memory before inference
    if torch.cuda.is_available():
        print(f"GPU memory before VGGT inference: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            
        # Log memory after aggregator
        if torch.cuda.is_available():
            print(f"GPU memory after aggregator: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        
        # Log memory after camera prediction
        if torch.cuda.is_available():
            print(f"GPU memory after camera prediction: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
        
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        
        # Log memory after depth prediction
        if torch.cuda.is_available():
            print(f"GPU memory after depth prediction: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    
    # Log memory after moving to CPU
    if torch.cuda.is_available():
        print(f"GPU memory after moving results to CPU: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
    
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    # Initialize GPU memory monitor
    gpu_monitor = GPUMemoryMonitor()
    print("GPU Memory Monitor initialized")
    gpu_monitor.log_memory_stats("at start")
    
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype: bfloat16 causes OOM error in my GPU (RTX 3090)
    # dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    dtype = torch.float16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    model = model.to(dtype)

    print(f"Model loaded")
    gpu_monitor.log_memory_stats("after model loading")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    
    # Load images with optional sorting
    if args.sort_images:
        print("Loading images with sorting (temporal order)")
        image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    else:
        print("Loading images without sorting (potentially better diversity)")
        image_path_list = glob.glob(os.path.join(image_dir, "*"))

    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # Select subset of images if specified
    image_path_list = select_images(
        image_path_list, 
        max_images=args.max_images, 
        method=args.image_selection_method
    )
    
    # Print first few images to show final selection
    print(f"Final selection: {len(image_path_list)} images")
    print("First 5 selected images:")
    for i, path in enumerate(image_path_list[:5]):
        print(f"  {i}: {os.path.basename(path)}")
    
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    vggt_fixed_resolution = args.vggt_resolution
    
    # Initialize variables for undistortion
    newK = None
    roi = None
    
    if args.load_imgs_squared:
        img_load_resolution = args.img_load_resolution
        print(f"Loading images with square padding and resize to {img_load_resolution}")
        print("Applying camera undistortion before processing" if args.undistort_images else "Loading without undistortion")
        images, original_coords, newK, roi, masks = load_and_preprocess_images_square(image_path_list, img_load_resolution, args.undistort_images)
        if args.undistort_images:
            print(f"Using optimal camera matrix for undistortion")
    else:
        print(f"Loading images without resizing (original resolution)")
        print("Applying camera undistortion to original resolution images" if args.undistort_images else "Loading without undistortion")
        images, original_coords, newK, roi, masks = load_and_preprocess_images_no_resize(image_path_list, args.undistort_images)
        if args.undistort_images:
            print(f"Using optimal camera matrix for undistortion")
        img_load_resolution = None  # Variable resolution
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir} with shape {images.shape}")
    gpu_monitor.log_memory_stats("after image loading")

    plt.figure()
    plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
    plt.savefig("sample_image.png")
    
    if masks is not None:
        plt.figure()
        plt.imshow(masks[0].permute(1, 2, 0).cpu().numpy())
        plt.savefig("sample_mask.png")
        print("Saved sample mask visualization")
    else:
        print("No masks available for visualization")
    
    # Run VGGT to estimate camera and depth
    # Use aspect ratio preservation when not using squared loading
    maintain_aspect_ratio = not args.load_imgs_squared
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution, maintain_aspect_ratio)
    # Inject known intrinsics for 3D point unprojection
    if args.use_known_intrinsics:
        # Use the camera matrix returned from loading function
        # This will be optimal matrix if undistortion was applied, otherwise original
        print(f"Using {'optimal' if args.undistort_images else 'original'} camera matrix from loading function as base intrinsics")
        original_known_intrinsics = newK
        
        # Get original and processed image dimensions
        original_width, original_height = original_coords[0, -2:].cpu().numpy()  # [1920, 1080]
        # TODO: Revise what happens if img loading resolution is not same as vggt_fixed_resolution(hence depth_map)
        processed_height, processed_width = images.shape[-2:]  # [1024, 1024] (padded square)
        # processed_height, processed_width = depth_map.shape[1], depth_map.shape[2] (952,952)
        
        print(f"Original image size: {original_height}x{original_width}")
        print(f"Processed image size: {processed_height}x{processed_width}")
        
        # Calculate padding transformation (same logic regardless of undistortion)
        if processed_height == processed_width:  # Square padding
            if original_width < original_height:
                # Width was padded to match height
                padding_x = (original_height - original_width) / 2
                padding_y = 0
            else:
                # Height was padded to match width
                padding_x = 0
                padding_y = (original_width - original_height) / 2
        else:
            # No padding, just resize
            padding_x = 0
            padding_y = 0
        
        print(f"Padding: x={padding_x}, y={padding_y}")
        
        # Apply padding transformation to intrinsics
        padded_intrinsics = original_known_intrinsics.copy()
        padded_intrinsics[0, 2] += padding_x  # cx += padding_x
        padded_intrinsics[1, 2] += padding_y  # cy += padding_y
        
        # Scale from padded size to processed size
        if original_width < original_height:
            # Padded to max(original_height) then resized to processed_width
            scale_factor = processed_width / original_height
        else:
            scale_factor = processed_height / original_width
        
        print(f"Original intrinsics:\n{original_known_intrinsics}")
        print(f"Padding adjusted intrinsics:\n{padded_intrinsics}")
        padded_intrinsics[:2, :] *= scale_factor  # Scale fx, fy, cx, cy
        print(f"Scale factor for intrinsics: {scale_factor}")
        print(f"Scaled padded intrinsics:\n{padded_intrinsics}")
        
        # Replicate for all frames
        intrinsic_for_unprojection = np.tile(padded_intrinsics[None, :, :], (len(images), 1, 1))
        
        # Finally, scale to match depth map resolution
        depth_height, depth_width = depth_map.shape[1], depth_map.shape[2]
        
        depth_scale_x = depth_width / processed_width
        depth_scale_y = depth_height / processed_height
        
        intrinsic_for_unprojection[:, 0, 0] *= depth_scale_x  # fx
        intrinsic_for_unprojection[:, 1, 1] *= depth_scale_y  # fy
        intrinsic_for_unprojection[:, 0, 2] *= depth_scale_x  # cx
        intrinsic_for_unprojection[:, 1, 2] *= depth_scale_y  # cy
        
        print(f"Final intrinsics for depth map ({depth_height}x{depth_width}):")
        print(intrinsic_for_unprojection[0])
        
        intrinsic = intrinsic_for_unprojection
    else:
        print("Using VGGT-predicted intrinsics for 3D point unprojection")
        intrinsic_for_unprojection = intrinsic
    
    if masks is not None:
        print("Using masks to filter background points in depth map unprojection")  
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic_for_unprojection, masks)
    
    # Apply same mask filtering to depth confidence
    if masks is not None:
        print("Applying mask filtering to depth confidence")
        depth_conf_masked = depth_conf.copy()
        for i in range(len(masks)):
            mask = masks[i]
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask[0]  # Remove channel dimension if present
            
            # Create foreground mask (values > 127 are foreground)
            foreground_mask = mask > 127
            # Zero out background confidence values
            depth_conf_masked[i][~foreground_mask] = 0.0
        
        depth_conf = depth_conf_masked
        print("Depth confidence filtered using masks")
    
    gpu_monitor.log_memory_stats("after VGGT inference")
    
    print(f"      Camera positions range: {np.min(extrinsic[:, :3, 3], axis=0)} to {np.max(extrinsic[:, :3, 3], axis=0)}")
    print(f"      Depth range: {np.min(depth_map):.3f} to {np.max(depth_map):.3f}")
    print(f"      Confidence range: {np.min(depth_conf):.3f} to {np.max(depth_conf):.3f}")

    del model
    torch.cuda.empty_cache()
    gpu_monitor.log_memory_stats("after model cleanup")

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera
        
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            images = images.to(device).to(dtype)
            # TODO: Masks are not used in the track prediction
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=masks,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()
            gpu_monitor.log_memory_stats("after track prediction")

        # rescale the intrinsic matrix from 518 to target resolution
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh
        print(f"Number of tracks after filtering: {np.sum(track_mask)}")

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)
        gpu_monitor.log_memory_stats("after bundle adjustment")

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = args.shared_camera  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        num_frames, height, width, _ = points_3d.shape
        image_size = np.array([height, width])
        points_rgb = F.interpolate(
            images, size=(height, width), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        print(f"Number of points before filtering: {depth_conf.shape}")
        conf_mask = depth_conf >= conf_thres_value
        print(f"Number of points after filtering: {np.sum(conf_mask)}")
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )
        gpu_monitor.log_memory_stats("after COLMAP conversion")

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
        use_known_intrinsics=args.use_known_intrinsics,
    )

    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/0/points.ply"))

    # Convert COLMAP model from binary to text format
    print("Converting COLMAP model to text format...")
    colmap_cmd = f"colmap model_converter --input_path {sparse_reconstruction_dir} --output_path {sparse_reconstruction_dir} --output_type TXT"
    print(f"Running: {colmap_cmd}")
    
    import subprocess
    try:
        result = subprocess.run(colmap_cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print("Successfully converted COLMAP model to text format")
        else:
            print(f"Warning: COLMAP conversion failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
    except Exception as e:
        print(f"Warning: Failed to run COLMAP converter: {e}")

    # Final memory logging
    gpu_monitor.log_memory_stats("at completion")
    
    # Log final memory summary
    peak_allocated, peak_reserved = gpu_monitor.get_peak_memory_mb(), gpu_monitor.get_peak_reserved_memory_mb()
    print(f"\n=== GPU Memory Summary ===")
    print(f"Peak Memory Allocated: {peak_allocated:.1f} MB")
    print(f"Peak Memory Reserved: {peak_reserved:.1f} MB")
    print(f"=========================")

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False, use_known_intrinsics=False
):
    rescale_camera = True
    
    # Define known intrinsics if needed
    known_intrinsics = np.array([
        [1543.5961, 0, 543.709494],
        [0, 1549.81553, 963.609549],
        [0, 0, 1]
    ])

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            
            if use_known_intrinsics:
                # Use known calibrated intrinsics as-is (no scaling or modification)
                print(f"Using known intrinsics for camera {pyimageid}")
                
                # Set camera parameters from known intrinsics (assuming PINHOLE model)
                pred_params = np.array([
                    known_intrinsics[0, 0],  # fx
                    known_intrinsics[1, 1],  # fy
                    known_intrinsics[0, 2],  # cx
                    known_intrinsics[1, 2]   # cy
                ])
            else:
                # Use predicted parameters with scaling
                pred_params = pred_params * resize_ratio
                real_pp = real_image_size / 2
                pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


def select_images(image_path_list, max_images=None, method="first"):
    """
    Select a subset of images from the full list based on the specified method.
    
    Args:
        image_path_list: List of all image paths
        max_images: Maximum number of images to select (None = use all)
        method: Selection method ("first", "last", "uniform")
        
    Returns:
        List of selected image paths
    """
    if max_images is None or max_images >= len(image_path_list):
        print(f"Using all {len(image_path_list)} images")
        return image_path_list
    
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    
    print(f"Selecting {max_images} images from {len(image_path_list)} total using method '{method}'")
    
    if method == "first":
        selected = image_path_list[:max_images]
    elif method == "last":
        selected = image_path_list[-max_images:]
    elif method == "uniform":
        # Select uniformly spaced images
        indices = np.linspace(0, len(image_path_list) - 1, max_images, dtype=int)
        selected = [image_path_list[i] for i in indices]
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    print(f"Selected images ({method} method):")
    for i, path in enumerate(selected[:10]):  # Show first 10
        print(f"  {i}: {os.path.basename(path)}")
    if len(selected) > 10:
        print(f"  ... and {len(selected) - 10} more images")
    
    return selected


# GPU Memory monitoring utilities
class GPUMemoryMonitor:
    def __init__(self):
        self.max_memory_allocated = 0
        self.max_memory_reserved = 0
        self.initial_memory = 0
        self.device_name = ""
        
        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name()
            self.initial_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
    
    def update_peak_memory(self):
        if torch.cuda.is_available():
            self.max_memory_allocated = max(self.max_memory_allocated, torch.cuda.max_memory_allocated())
            self.max_memory_reserved = max(self.max_memory_reserved, torch.cuda.max_memory_reserved())
    
    def get_current_memory_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    def get_peak_memory_mb(self):
        self.update_peak_memory()
        return self.max_memory_allocated / 1024 / 1024
    
    def get_peak_reserved_memory_mb(self):
        self.update_peak_memory()
        return self.max_memory_reserved / 1024 / 1024
    
    def log_memory_stats(self, stage=""):
        if torch.cuda.is_available():
            current_mb = self.get_current_memory_mb()
            peak_mb = self.get_peak_memory_mb()
            reserved_mb = self.get_peak_reserved_memory_mb()
            
            print(f"GPU Memory {stage}:")
            print(f"  Device: {self.device_name}")
            print(f"  Current: {current_mb:.1f} MB")
            print(f"  Peak Allocated: {peak_mb:.1f} MB")
            print(f"  Peak Reserved: {reserved_mb:.1f} MB")
            return peak_mb, reserved_mb
        else:
            print(f"GPU Memory {stage}: CUDA not available")
            return 0, 0


if __name__ == "__main__":
    import os
    import sys
    from datetime import datetime
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    class Args:
        def __init__(self):
            self.scene_dir = "vggt-hh/dataset_vggt/Ob68/handheld"  # Update this path
            self.seed = 42
            self.camera_type = "PINHOLE"

            # Image selection parameters
            self.max_images = 20  # Limit number of images
            self.image_selection_method = "first"  # Options: first, last, uniform
            
            # BA parameters
            self.use_ba = True
            self.max_reproj_error = 8.0
            self.vis_thresh = 0.2
            self.query_frame_num = self.max_images
            self.max_query_pts = 4096
            self.fine_tracking = False
            
            # Non-BA parameters
            self.conf_thres_value = 5.0
            
            # Image loading parameters
            self.shared_camera = True
            self.sort_images = True
            self.use_known_intrinsics = True  # Use known calibrated intrinsics
            self.load_imgs_squared = True  # Use square padding and resize
            self.img_load_resolution = 952  # Target resolution when using load_imgs_squared
            self.vggt_resolution = 952  # VGGT model resolution
            self.undistort_images = True  # Undistort images using known camera intrinsics
            

    # For 24GB GPU:
    # With squared 518x518 -> max_images=117
    # With squared 700x700 -> max_images=61
    # With squared 952x952 -> max_images=28
    # With non-squared 952x532 -> max_images=62

    args = Args()
    
    # Set up logging
    sparse_dir = os.path.join(args.scene_dir, "sparse/0")
    os.makedirs(sparse_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(sparse_dir, f"reconstruction_log_{timestamp}.txt")
    
    # Simple tee class to write to both console and file
    class Tee:
        def __init__(self, file):
            self.file = file
            self.stdout = sys.stdout
        def write(self, data):
            self.file.write(data)
            self.file.flush()
            self.stdout.write(data)
        def flush(self):
            self.file.flush()
            self.stdout.flush()
    
    # Execute with logging
    with open(log_file, 'w') as f:
        f.write(f"VGGT Reconstruction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Scene: {args.scene_dir}\n")
        f.write(f"Image Selection: {args.max_images if args.max_images else 'all'} images using '{args.image_selection_method}' method\n")
        f.write("="*60 + "\n\n")
        
        original_stdout = sys.stdout
        sys.stdout = Tee(f)
        
        try:
            with torch.no_grad():
                result = demo_fn(args)
            print(f"\nLog saved to: {log_file}")
        finally:
            sys.stdout = original_stdout


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
