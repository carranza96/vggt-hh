def umeyama_ransac(src_points, dst_points, threshold=0.05, max_trials=100, min_inliers=3, with_scale=True):
    """Robust Umeyama alignment using RANSAC on point correspondences."""
    assert src_points.shape == dst_points.shape and src_points.shape[1] == 3
    n = src_points.shape[0]
    best_inliers = []
    best_R, best_t, best_scale = None, None, None
    rng = np.random.default_rng()
    for _ in range(max_trials):
        # Randomly sample minimal set (3 for 3D similarity)
        idx = rng.choice(n, 3, replace=False)
        R, t, scale = umeyama_sim_transform(src_points[idx], dst_points[idx], with_scale=with_scale)
        # Transform all src_points
        src_aligned = (scale * (R @ src_points.T)).T + t
        errors = np.linalg.norm(src_aligned - dst_points, axis=1)
        inliers = np.where(errors < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R, best_t, best_scale = R, t, scale
        # Early exit if enough inliers
        if len(best_inliers) >= n * 0.8:
            break
    if len(best_inliers) >= min_inliers:
        # Re-estimate using all inliers
        R, t, scale = umeyama_sim_transform(src_points[best_inliers], dst_points[best_inliers], with_scale=with_scale)
        return R, t, scale, best_inliers
    else:
        # Fallback to all points
        R, t, scale = umeyama_sim_transform(src_points, dst_points, with_scale=with_scale)
        return R, t, scale, np.arange(n)

import pickle
import hashlib

def compute_chunk_cache_key(chunk_paths, args):
    """Compute a unique cache key for a chunk based on image names and key args."""
    # Use image basenames and key parameters
    key = {
        'images': [os.path.basename(p) for p in chunk_paths],
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'use_ba_per_chunk': args.use_ba_per_chunk,
        'img_load_resolution': args.img_load_resolution,
        'vggt_resolution': args.vggt_resolution,
        'shared_camera': args.shared_camera,
        'use_known_intrinsics': args.use_known_intrinsics,
        'undistort_images': args.undistort_images,
    }
    # Use a stable hash (SHA256) for consistent cache keys across runs
    key_bytes = pickle.dumps(key)
    key_hash = hashlib.sha256(key_bytes).hexdigest()
    return key_hash

def get_chunk_cache_dir(args):
    """Return the directory where chunk caches are stored."""
    cache_dir = os.path.join(args.scene_dir, "chunk_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def save_chunk_result_to_cache(chunk_result, chunk_cache_path):
    # Save the reconstruction using pycolmap (if present)
    recon = chunk_result.get('reconstruction', None)
    if recon is not None:
        recon_dir = chunk_cache_path + "_recon"
        os.makedirs(recon_dir, exist_ok=True)
        recon.write(recon_dir)
        # Remove from dict to avoid pickle issues
        chunk_result = dict(chunk_result)
        chunk_result['reconstruction'] = None
        chunk_result['reconstruction_dir'] = recon_dir
    with open(chunk_cache_path, 'wb'):
        pickle.dump(chunk_result, open(chunk_cache_path, 'wb'))

def load_chunk_result_from_cache(chunk_cache_path):
    with open(chunk_cache_path, 'rb') as f:
        chunk_result = pickle.load(f)
    # Load reconstruction if present
    recon_dir = chunk_result.get('reconstruction_dir', None)
    if recon_dir and os.path.exists(recon_dir):
        try:
            recon = pycolmap.Reconstruction()
            recon.read(recon_dir)
            chunk_result['reconstruction'] = recon
        except Exception as e:
            print(f"Warning: Failed to load reconstruction from {recon_dir}: {e}")
            chunk_result['reconstruction'] = None
    return chunk_result
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
import traceback
import subprocess
from scipy.spatial.transform import Rotation

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
from utils_hh import GPUMemoryMonitor, select_images, Tee
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Chunked Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    
    ######### Image loading parameters #########
    parser.add_argument("--sort_images", action="store_true", default=False, 
                       help="Sort images by filename (default: unsorted for better diversity)")
    parser.add_argument("--use_known_intrinsics", action="store_true", default=False,
                       help="Use known calibrated intrinsics instead of VGGT predictions")
    parser.add_argument("--load_imgs_squared", action="store_true", default=True,
                       help="Load images with square padding and resizing (default: True)")
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
    
    ######### Chunking parameters #########
    parser.add_argument("--chunk_size", type=int, default=10,
                       help="Number of images per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=3,
                       help="Number of overlapping images between chunks")
    parser.add_argument("--use_ba_per_chunk", action="store_true", default=False,
                       help="Run bundle adjustment on each chunk")
    parser.add_argument("--save_chunk_debug", action="store_true", default=False,
                       help="Save individual chunk reconstructions for debugging")
    
    ######### BA parameters #########
    parser.add_argument("--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction")
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument("--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)")
    parser.add_argument("--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)")
    
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518, maintain_aspect_ratio=False):
    assert len(images.shape) == 4 and images.shape[1] == 3

    if maintain_aspect_ratio:
        B, C, H, W = images.shape
        if H >= W:
            new_H, new_W = resolution, int(W * (resolution / H))
        else:
            new_W, new_H = resolution, int(H * (resolution / W))
        new_H, new_W = (new_H // 14) * 14, (new_W // 14) * 14
        images = F.interpolate(images, size=(new_H, new_W), mode="bilinear", align_corners=False)
    else:
        images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    return (extrinsic.squeeze(0).cpu().numpy(), intrinsic.squeeze(0).cpu().numpy(), 
            depth_map.squeeze(0).cpu().numpy(), depth_conf.squeeze(0).cpu().numpy())


def inject_known_intrinsics(newK, original_coords, images_shape, depth_shape, 
                           load_imgs_squared, img_load_resolution, num_images):
    """Helper function to inject known intrinsics with proper scaling."""
    original_width, original_height = original_coords[0, -2:].cpu().numpy()
    processed_height, processed_width = images_shape[-2:]
    
    # Calculate padding transformation
    if load_imgs_squared and processed_height == processed_width:
        if original_width < original_height:
            padding_x, padding_y = (original_height - original_width) / 2, 0
        else:
            padding_x, padding_y = 0, (original_width - original_height) / 2
    else:
        padding_x = padding_y = 0
    
    # Apply padding and scaling transformations
    padded_intrinsics = newK.copy()
    padded_intrinsics[0, 2] += padding_x
    padded_intrinsics[1, 2] += padding_y
    
    if load_imgs_squared:
        scale_factor = processed_width / original_height if original_width < original_height else processed_height / original_width
    else:
        scale_factor = min(processed_width / original_width, processed_height / original_height)
    
    padded_intrinsics[:2, :] *= scale_factor
    intrinsic_for_unprojection = np.tile(padded_intrinsics[None, :, :], (num_images, 1, 1))
    
    # Scale to match depth map resolution
    depth_height, depth_width = depth_shape[1], depth_shape[2]
    depth_scale_x, depth_scale_y = depth_width / processed_width, depth_height / processed_height
    
    intrinsic_for_unprojection[:, 0, 0] *= depth_scale_x
    intrinsic_for_unprojection[:, 1, 1] *= depth_scale_y
    intrinsic_for_unprojection[:, 0, 2] *= depth_scale_x
    intrinsic_for_unprojection[:, 1, 2] *= depth_scale_y
    
    return intrinsic_for_unprojection


def create_image_chunks(image_path_list, chunk_size, overlap):
    """Create overlapping chunks of images for processing."""
    chunks = []
    step = chunk_size - overlap
    
    if len(image_path_list) < chunk_size:
        return [list(zip(image_path_list, range(len(image_path_list))))]
    
    start_idx = 0
    while True:
        end_idx = min(start_idx + chunk_size, len(image_path_list))
        if end_idx < start_idx + chunk_size and start_idx > 0:
            start_idx = len(image_path_list) - chunk_size
            end_idx = len(image_path_list)
        
        chunk_paths = image_path_list[start_idx:end_idx]
        chunk_indices = list(range(start_idx, end_idx))
        chunks.append(list(zip(chunk_paths, chunk_indices)))
        
        if end_idx == len(image_path_list):
            break
        start_idx += step
    
    return chunks


def process_chunk(model, chunk_data, args, dtype, gpu_monitor, chunk_idx, newK=None):
    """Process a single chunk of images and return reconstruction data."""
    chunk_paths = [item[0] for item in chunk_data]
    global_indices = [item[1] for item in chunk_data]

    # Persistent chunk cache
    cache_dir = get_chunk_cache_dir(args)
    cache_key = compute_chunk_cache_key(chunk_paths, args)
    chunk_cache_path = os.path.join(cache_dir, f"chunk_{chunk_idx:03d}_{cache_key}.pkl")

    if os.path.exists(chunk_cache_path):
        print(f"\n=== Loading Chunk {chunk_idx + 1} from cache ===")
        return load_chunk_result_from_cache(chunk_cache_path)

    print(f"\n=== Processing Chunk {chunk_idx + 1} ===")
    print(f"Images {global_indices[0]} to {global_indices[-1]} ({len(chunk_paths)} images)")

    if args.load_imgs_squared:
        images, original_coords, chunk_newK, chunk_roi, masks = load_and_preprocess_images_square(
            chunk_paths, args.img_load_resolution, args.undistort_images)
        img_load_resolution = args.img_load_resolution
    else:
        images, original_coords, chunk_newK, chunk_roi, masks = load_and_preprocess_images_no_resize(
            chunk_paths, args.undistort_images)
        img_load_resolution = None

    original_coords = original_coords.to(next(model.parameters()).device)

    maintain_aspect_ratio = not args.load_imgs_squared
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(
        model, images, dtype, args.vggt_resolution, maintain_aspect_ratio)

    if args.use_known_intrinsics and newK is not None:
        intrinsic = inject_known_intrinsics(
            newK, original_coords, images.shape, depth_map.shape,
            args.load_imgs_squared, img_load_resolution, len(images))

    if masks is not None:
        print("Using masks to filter background points")
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic, masks)

    if masks is not None:
        depth_conf_masked = depth_conf.copy()
        for i in range(len(masks)):
            mask = masks[i]
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask[0]
            foreground_mask = mask > 127
            depth_conf_masked[i][~foreground_mask] = 0.0
        depth_conf = depth_conf_masked

    reconstruction = None

    if args.use_ba or args.use_ba_per_chunk:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / args.vggt_resolution if img_load_resolution else 1.0

        with torch.cuda.amp.autocast(dtype=dtype):
            images = images.to(next(model.parameters()).device).to(dtype)
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images, conf=depth_conf, points_3d=points_3d, masks=masks,
                max_query_pts=args.max_query_pts, query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp", fine_tracking=args.fine_tracking)

        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d, extrinsic, intrinsic, pred_tracks, image_size, masks=track_mask,
            max_reproj_error=args.max_reproj_error, shared_camera=args.shared_camera,
            camera_type=args.camera_type, points_rgb=points_rgb)

        if reconstruction is not None and args.use_ba_per_chunk:
            ba_options = pycolmap.BundleAdjustmentOptions(refine_focal_length=False,
            refine_principal_point=False,
            refine_extra_params=False)
            pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        num_frames, height, width, _ = points_3d.shape
        image_size = np.array([height, width])
        points_rgb = F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
        conf_mask = depth_conf >= args.conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, 100000)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d, points_xyf, points_rgb, extrinsic, intrinsic, image_size,
            shared_camera=args.shared_camera, camera_type="PINHOLE")

        reconstruction_resolution = args.vggt_resolution

    base_image_path_list = [os.path.basename(path) for path in chunk_paths]

    if reconstruction is not None:
        reconstruction = rename_colmap_recons_and_rescale_camera(
            reconstruction, base_image_path_list, original_coords.cpu().numpy(),
            img_size=reconstruction_resolution, shift_point2d_to_original_res=True,
            shared_camera=args.shared_camera, use_known_intrinsics=args.use_known_intrinsics)

    # Always save individual chunk reconstructions before alignment
    if reconstruction is not None:
        chunk_dir = os.path.join(args.scene_dir, f"chunks/chunk_{chunk_idx:03d}_before_alignment")
        os.makedirs(chunk_dir, exist_ok=True)
        sparse_dir = os.path.join(chunk_dir, "sparse/0")
        os.makedirs(sparse_dir, exist_ok=True)

        reconstruction.write(sparse_dir)

        try:
            text_cmd = f"colmap model_converter --input_path {sparse_dir} --output_path {sparse_dir} --output_type TXT"
            result = subprocess.run(text_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Saved chunk {chunk_idx + 1} (before alignment) in text format")
            else:
                print(f"  Warning: Failed to convert chunk {chunk_idx + 1} to text format")
        except Exception as e:
            print(f"  Warning: COLMAP text conversion failed for chunk {chunk_idx + 1}: {e}")

        if len(points_3d.shape) == 4:
            flat_points = points_3d.reshape(-1, 3)
            flat_colors = points_rgb.reshape(-1, 3) if len(points_rgb.shape) == 4 else points_rgb
        else:
            flat_points, flat_colors = points_3d, points_rgb

        trimesh.PointCloud(flat_points, colors=flat_colors).export(
            os.path.join(sparse_dir, "points.ply"))

        print(f"  Saved chunk {chunk_idx + 1} reconstruction (before alignment) to {chunk_dir}")

    # Save chunk result to cache (after all processing, including BA)
    chunk_result = {
        'reconstruction': reconstruction,
        'points_3d': points_3d,
        'points_rgb': points_rgb,
        'chunk_paths': chunk_paths,
        'global_indices': global_indices
    }
    save_chunk_result_to_cache(chunk_result, chunk_cache_path)
    return chunk_result


def umeyama_sim_transform(src, dst, with_scale=True):
    """Estimate similarity transform (s,R,t) to align src to dst."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n, m = src.shape
    
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_centered, dst_centered = src - mu_src, dst - mu_dst
    
    cov_matrix = dst_centered.T @ src_centered / n
    U, D, Vt = np.linalg.svd(cov_matrix)
    
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    
    R = U @ S @ Vt
    
    if with_scale:
        var_src = (src_centered ** 2).sum() / n
        scale = np.trace(np.diag(D) @ S) / var_src
    else:
        scale = 1.0
    
    t = mu_dst - scale * R @ mu_src
    return R, t, scale


def apply_similarity_transform_to_reconstruction(reconstruction, scale, R, t):
    """Apply similarity transformation to a reconstruction."""
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    
    for image_id, image in sorted(reconstruction.images.items()):
        if image.registered:
            rigid3d = image.cam_from_world
            quat_xyzw = rigid3d.rotation.quat
            tvec = rigid3d.translation
            R_cam = Rotation.from_quat(quat_xyzw).as_matrix()
            
            camera_center = -R_cam.T @ tvec
            camera_center_homo = np.append(camera_center, 1.0)
            new_camera_center = (T @ camera_center_homo)[:3]
            
            new_R_cam = R_cam @ R.T
            new_t_cam = -new_R_cam @ new_camera_center
            new_rot = Rotation.from_matrix(new_R_cam)
            new_quat_xyzw = new_rot.as_quat()
            
            image.cam_from_world = pycolmap.Rigid3d(new_quat_xyzw, new_t_cam)
    
    for point_id, point in reconstruction.points3D.items():
        old_point = np.append(point.xyz, 1.0)
        new_point = T @ old_point
        point.xyz = new_point[:3]


def save_aligned_chunk(reconstruction, chunk_idx, args):
    """Save a chunk reconstruction after alignment in both binary and text format."""
    chunk_dir = os.path.join(args.scene_dir, f"chunks/chunk_{chunk_idx:03d}_after_alignment")
    os.makedirs(chunk_dir, exist_ok=True)
    sparse_dir = os.path.join(chunk_dir, "sparse/0")
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Save in binary format
    reconstruction.write(sparse_dir)
    
    # Convert to text format
    try:
        text_cmd = f"colmap model_converter --input_path {sparse_dir} --output_path {sparse_dir} --output_type TXT"
        result = subprocess.run(text_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    Saved aligned chunk {chunk_idx + 1} in text format")
        else:
            print(f"    Warning: Failed to convert aligned chunk {chunk_idx + 1} to text format")
    except Exception as e:
        print(f"    Warning: COLMAP text conversion failed for aligned chunk {chunk_idx + 1}: {e}")
    
    print(f"    Saved aligned chunk {chunk_idx + 1} to {chunk_dir}")


def save_intermediate_merge(reconstruction, up_to_chunk_idx, args):
    """Save intermediate merged reconstruction after adding each chunk."""
    merge_dir = os.path.join(args.scene_dir, f"merging_steps/merged_up_to_chunk_{up_to_chunk_idx:03d}")
    os.makedirs(merge_dir, exist_ok=True)
    sparse_dir = os.path.join(merge_dir, "sparse/0")
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Save in binary format
    reconstruction.write(sparse_dir)
    
    # Convert to text format
    try:
        text_cmd = f"colmap model_converter --input_path {sparse_dir} --output_path {sparse_dir} --output_type TXT"
        result = subprocess.run(text_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    Saved intermediate merge (up to chunk {up_to_chunk_idx + 1}) in text format")
        else:
            print(f"    Warning: Failed to convert intermediate merge to text format")
    except Exception as e:
        print(f"    Warning: COLMAP text conversion failed for intermediate merge: {e}")
    
    # Save summary info
    with open(os.path.join(merge_dir, "merge_info.txt"), 'w') as f:
        f.write(f"Intermediate Merge - Up to Chunk {up_to_chunk_idx + 1}\n")
        f.write("="*50 + "\n")
        f.write(f"Cameras: {len(reconstruction.cameras)}\n")
        f.write(f"Images: {len(reconstruction.images)}\n")
        f.write(f"Points: {len(reconstruction.points3D)}\n")
        f.write(f"Chunks merged: 0 to {up_to_chunk_idx}\n\n")
        
        f.write("Image list:\n")
        for img_id, img in sorted(reconstruction.images.items()):
            f.write(f"  {img_id}: {img.name}\n")
    
    print(f"    Saved intermediate merge (up to chunk {up_to_chunk_idx + 1}) to {merge_dir}")


def simple_concatenate_reconstructions(base_reconstruction, src_reconstruction, args):
    """Simple concatenation of reconstructions with proper ID management."""
    merged = copy.deepcopy(base_reconstruction)
    
    next_camera_id = max(merged.cameras.keys()) + 1 if merged.cameras else 1
    next_image_id = max(merged.images.keys()) + 1 if merged.images else 1
    
    # Camera ID mapping
    camera_id_map = {}
    for src_camera_id, src_camera in src_reconstruction.cameras.items():
        if args.shared_camera and len(merged.cameras) > 0:
            camera_id_map[src_camera_id] = list(merged.cameras.keys())[0]
        else:
            camera_id_map[src_camera_id] = next_camera_id
            try:
                new_camera = pycolmap.Camera(
                    camera_id=next_camera_id, model_name=src_camera.model_name,
                    width=src_camera.width, height=src_camera.height, params=src_camera.params)
                merged.add_camera(new_camera)
                next_camera_id += 1
            except Exception:
                if len(merged.cameras) > 0:
                    camera_id_map[src_camera_id] = list(merged.cameras.keys())[0]
    
    # Add images (skip duplicates)
    image_id_map = {}
    for src_image_id, src_image in sorted(src_reconstruction.images.items()):
        image_exists = any(img.name == src_image.name for img in merged.images.values())
        if image_exists:
            continue
        
        new_image_id = next_image_id
        image_id_map[src_image_id] = new_image_id
        new_camera_id = camera_id_map.get(src_image.camera_id, 1)
        
        try:
            new_image = pycolmap.Image(
                image_id=new_image_id, name=src_image.name, camera_id=new_camera_id,
                cam_from_world=src_image.cam_from_world)
            
            for point2d in src_image.points2D:
                new_image.points2D.append(pycolmap.Point2D(xy=point2d.xy))
            
            merged.add_image(new_image)
            next_image_id += 1
        except Exception:
            continue
    
    # Add 3D points with proper ID mapping
    for src_point_id, src_point in sorted(src_reconstruction.points3D.items()):
        try:
            new_track = pycolmap.Track()
            for track_element in src_point.track.elements:
                if track_element.image_id in image_id_map:
                    new_image_id = image_id_map[track_element.image_id]
                    new_track.add_element(new_image_id, track_element.point2D_idx)
            
            if len(new_track.elements) > 0:
                merged.add_point3D(xyz=src_point.xyz, track=new_track, color=src_point.color)
        except Exception:
            continue
    
    # Register newly added images
    for src_image_id, new_image_id in image_id_map.items():
        try:
            src_image = src_reconstruction.images[src_image_id]
            if src_image.registered:
                merged.register_image(new_image_id)
        except Exception:
            continue
    
    return merged


def merge_chunk_reconstructions(chunk_results, args):
    """Simple merging using Umeyama alignment for overlapping chunks."""
    valid_chunks = [chunk for chunk in chunk_results if chunk['reconstruction'] is not None]
    
    if len(valid_chunks) == 0:
        return None
    if len(valid_chunks) == 1:
        return copy.deepcopy(valid_chunks[0]['reconstruction'])
    
    print(f"Merging {len(valid_chunks)} valid chunks with Umeyama alignment")
    merged_reconstruction = copy.deepcopy(valid_chunks[0]['reconstruction'])
    
    # Save the first chunk as reference (no alignment needed)
    save_aligned_chunk(merged_reconstruction, 0, args)
    
    # Save initial state (first chunk only) as merge step 0
    save_intermediate_merge(merged_reconstruction, 0, args)
    
    for chunk_idx in range(1, len(valid_chunks)):
        src_reconstruction = copy.deepcopy(valid_chunks[chunk_idx]['reconstruction'])
        
        # Find common images for alignment
        common_image_pairs = []
        for src_img_id, src_image in src_reconstruction.images.items():
            for merged_img_id, merged_image in merged_reconstruction.images.items():
                if src_image.name == merged_image.name:
                    common_image_pairs.append((src_img_id, merged_img_id))
        
        print(f"  Found {len(common_image_pairs)} common images")
        
        # Align using visual tracks (2D-3D correspondences) if sufficient overlap
        src_points = []
        ref_points = []
        # For each common image, match points2D by pixel location
        for src_img_id, merged_img_id in sorted(common_image_pairs):
            src_image = src_reconstruction.images[src_img_id]
            merged_image = merged_reconstruction.images[merged_img_id]
            # Build dicts: pixel -> point3D_id
            src_xy_to_3d = {tuple(np.round(p2d.xy, 2)): p2d.point3D_id for p2d in src_image.points2D if p2d.has_point3D()}
            ref_xy_to_3d = {tuple(np.round(p2d.xy, 2)): p2d.point3D_id for p2d in merged_image.points2D if p2d.has_point3D()}
            # Find common pixels
            common_pixels = set(src_xy_to_3d.keys()) & set(ref_xy_to_3d.keys())
            for pix in common_pixels:
                src_pid = src_xy_to_3d[pix]
                ref_pid = ref_xy_to_3d[pix]
                # Get 3D coordinates
                src_pt = src_reconstruction.points3D[src_pid].xyz
                ref_pt = merged_reconstruction.points3D[ref_pid].xyz
                src_points.append(src_pt)
                ref_points.append(ref_pt)
        src_points = np.array(src_points)
        ref_points = np.array(ref_points)
        print(f"  Found {len(src_points)} 2D-3D correspondences for alignment")
        if len(src_points) >= 3:
            # Use RANSAC for robust alignment
            R, t, scale, inliers = umeyama_ransac(src_points, ref_points, threshold=0.05, max_trials=100, min_inliers=3, with_scale=True)
            print(f"Scale: {scale:.4f}, Rotation:\n{R}\nTranslation: {t}")
            print(f"Inliers: {len(inliers)}/{len(src_points)}")

            # R, t, scale = umeyama_sim_transform(src_points, ref_points, with_scale=True)
            # print(f"Scale: {scale:.4f}, Rotation:\n{R}\nTranslation: {t}")

            # Error statistics
            src_points_aligned = (scale * (R @ src_points.T)).T + t
            errors = np.linalg.norm(src_points_aligned - ref_points, axis=1)
            print("Alignment errors (per point):", errors)
            print("Alignment RMS error:", np.sqrt(np.mean(errors**2)))

            apply_similarity_transform_to_reconstruction(src_reconstruction, scale, R, t)
            save_aligned_chunk(src_reconstruction, chunk_idx, args)
        else:
            # Save unaligned chunk (insufficient overlap)
            save_aligned_chunk(src_reconstruction, chunk_idx, args)
        
        merged_reconstruction = simple_concatenate_reconstructions(
            merged_reconstruction, src_reconstruction, args)
        
        # Save intermediate merged reconstruction after adding this chunk
        save_intermediate_merge(merged_reconstruction, chunk_idx, args)
        
        
        # print("Running intermediate bundle adjustment...")
        ba_options = pycolmap.BundleAdjustmentOptions(refine_focal_length=False,
                    refine_principal_point=False,
                    refine_extra_params=False)
        pycolmap.bundle_adjustment(merged_reconstruction, ba_options)
    
    return merged_reconstruction


def rename_colmap_recons_and_rescale_camera(reconstruction, image_paths, original_coords, img_size, 
                                          shift_point2d_to_original_res=False, shared_camera=False, 
                                          use_known_intrinsics=False):
    """Rename images and rescale camera parameters."""
    rescale_camera = True
    known_intrinsics = np.array([[1543.5961, 0, 543.709494], [0, 1549.81553, 963.609549], [0, 0, 1]])

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            
            if use_known_intrinsics:
                pred_params = np.array([known_intrinsics[0, 0], known_intrinsics[1, 1], 
                                      known_intrinsics[0, 2], known_intrinsics[1, 2]])
            else:
                pred_params = pred_params * resize_ratio
                pred_params[-2:] = real_image_size / 2

            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


def demo_fn_chunked(args):
    """Main demo function that processes images in chunks."""
    # Initialize GPU memory monitor
    gpu_monitor = GPUMemoryMonitor()
    print("GPU Memory Monitor initialized")
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Set device and dtype
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")

    # Load VGGT model
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval().to(device).to(dtype)

    # Get image paths
    image_dir = os.path.join(args.scene_dir, "images")
    if args.sort_images:
        image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    else:
        image_path_list = glob.glob(os.path.join(image_dir, "*"))

    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # Select subset of images if specified
    image_path_list = select_images(image_path_list, max_images=args.max_images, 
                                  method=args.image_selection_method)
    
    print(f"Total images to process: {len(image_path_list)}")
    
    # Get known intrinsics if needed
    newK = None
    if args.use_known_intrinsics or args.undistort_images:
        # Load a sample image to get camera matrix
        if args.load_imgs_squared:
            _, _, newK, _, _ = load_and_preprocess_images_square([image_path_list[0]], 
                                                              args.img_load_resolution, args.undistort_images)
        else:
            _, _, newK, _, _ = load_and_preprocess_images_no_resize([image_path_list[0]], args.undistort_images)
    
    # Create chunks
    chunks = create_image_chunks(image_path_list, args.chunk_size, args.chunk_overlap)
    print(f"Created {len(chunks)} chunks for processing")
    
    # Process each chunk
    chunk_results = []
    for chunk_idx, chunk_data in enumerate(chunks):
        try:
            chunk_result = process_chunk(model, chunk_data, args, dtype, gpu_monitor, chunk_idx, newK)
            chunk_results.append(chunk_result)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing chunk {chunk_idx + 1}: {e}")
            chunk_results.append({'reconstruction': None, 'chunk_paths': [item[0] for item in chunk_data],
                                'global_indices': [item[1] for item in chunk_data], 'error': str(e)})
    
    # Clean up model
    del model
    torch.cuda.empty_cache()
    
    # Merge chunk reconstructions
    print(f"\n=== Merging Chunk Reconstructions ===")
    merged_reconstruction = merge_chunk_reconstructions(chunk_results, args)
    
    if merged_reconstruction is None:
        print("Failed to create merged reconstruction!")
        return False
    
    # Final bundle adjustment
    if args.use_ba:
        print("Running final bundle adjustment...")
        ba_options = pycolmap.BundleAdjustmentOptions(refine_focal_length=False,
            refine_principal_point=False,
            refine_extra_params=False)
        pycolmap.bundle_adjustment(merged_reconstruction, ba_options)
    
    # Save final reconstruction
    print(f"Saving final reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    merged_reconstruction.write(sparse_reconstruction_dir)
    
    # Save point cloud
    all_points_3d, all_points_rgb = [], []
    for chunk_result in chunk_results:
        if chunk_result['reconstruction'] is not None:
            points_3d = chunk_result['points_3d']
            points_rgb = chunk_result['points_rgb']
            
            if len(points_3d.shape) == 4:
                points_3d = points_3d.reshape(-1, 3)
                points_rgb = points_rgb.reshape(-1, 3) if len(points_rgb.shape) == 4 else points_rgb
            
            all_points_3d.append(points_3d)
            all_points_rgb.append(points_rgb)
    
    if all_points_3d:
        final_points_3d = np.concatenate(all_points_3d, axis=0)
        final_points_rgb = np.concatenate(all_points_rgb, axis=0)
        trimesh.PointCloud(final_points_3d, colors=final_points_rgb).export(
            os.path.join(args.scene_dir, "sparse/0/points.ply"))
    
    # Convert COLMAP model to text format
    colmap_cmd = f"colmap model_converter --input_path {sparse_reconstruction_dir} --output_path {sparse_reconstruction_dir} --output_type TXT"
    try:
        result = subprocess.run(colmap_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully converted COLMAP model to text format")
    except Exception as e:
        print(f"Warning: Failed to run COLMAP converter: {e}")

    print(f"Final reconstruction: {len(merged_reconstruction.cameras)} cameras, "
          f"{len(merged_reconstruction.images)} images, {len(merged_reconstruction.points3D)} points")
    
    # Print summary of saved chunks
    print(f"\n=== Chunk Saving Summary ===")
    chunks_dir = os.path.join(args.scene_dir, "chunks")
    merging_dir = os.path.join(args.scene_dir, "merging_steps")
    
    if os.path.exists(chunks_dir):
        chunk_dirs = [d for d in os.listdir(chunks_dir) if os.path.isdir(os.path.join(chunks_dir, d))]
        before_dirs = [d for d in chunk_dirs if "before_alignment" in d]
        after_dirs = [d for d in chunk_dirs if "after_alignment" in d]
        debug_dirs = [d for d in chunk_dirs if "debug" in d]
        
        print(f"Saved {len(before_dirs)} chunks before alignment")
        print(f"Saved {len(after_dirs)} chunks after alignment")
        if debug_dirs:
            print(f"Saved {len(debug_dirs)} debug chunks")
        print(f"All chunks saved in: {chunks_dir}")
    
    if os.path.exists(merging_dir):
        merge_dirs = [d for d in os.listdir(merging_dir) if os.path.isdir(os.path.join(merging_dir, d))]
        print(f"Saved {len(merge_dirs)} intermediate merging steps")
        print(f"Merging steps saved in: {merging_dir}")
    
    return True


if __name__ == "__main__":
    import os
    import sys
    from datetime import datetime
    
    class Args:
        def __init__(self):
            self.scene_dir = "vggt-hh/dataset_vggt/Ob68/handheld"
            self.seed = 42
            self.camera_type = "PINHOLE"
            
            # Chunking parameters
            self.chunk_size = 15
            self.chunk_overlap = 5
            self.use_ba_per_chunk = True
            self.save_chunk_debug = False
            
            # Image selection parameters
            self.max_images = 129
            self.image_selection_method = "first"
            
            # BA parameters
            self.use_ba = False
            self.max_reproj_error = 8.0
            self.vis_thresh = 0.2
            self.query_frame_num = self.max_images
            self.max_query_pts = 4096
            self.fine_tracking = False
            self.conf_thres_value = 5.0
            
            # Image loading parameters
            self.shared_camera = True
            self.sort_images = True
            self.use_known_intrinsics = True
            self.load_imgs_squared = True
            self.img_load_resolution = 952
            self.vggt_resolution = 952
            self.undistort_images = True

    args = Args()
    
    # Set up logging
    sparse_dir = os.path.join(args.scene_dir, "sparse/0")
    os.makedirs(sparse_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(sparse_dir, f"chunked_reconstruction_log_{timestamp}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"VGGT Chunked Reconstruction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Scene: {args.scene_dir}\n")
        f.write(f"Chunking: {args.chunk_size} images per chunk, {args.chunk_overlap} overlap\n")
        f.write("="*60 + "\n\n")
        
        original_stdout = sys.stdout
        sys.stdout = Tee(f)
        
        try:
            with torch.no_grad():
                result = demo_fn_chunked(args)
            print(f"\nLog saved to: {log_file}")
        finally:
            sys.stdout = original_stdout


"""
VGGT Chunked Runner Script
=========================

A clean, streamlined script for chunked VGGT 3D reconstruction.

Key Features:
• Chunked Processing: Process large image sets in smaller chunks
• Umeyama Alignment: Align overlapping chunks using similarity transforms
• Memory Efficient: Processes chunks sequentially to manage GPU memory
• COLMAP Compatible: Outputs standard COLMAP sparse reconstruction format
"""
