# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import cv2


def get_camera_intrinsics_for_undistortion():
    """
    Get the original camera matrix and distortion coefficients for undistortion.
    
    Returns:
        tuple: (camera_matrix, dist_coeffs) as numpy arrays
    """
    camera_matrix = np.array([
        [1.54359610e+03, 0.00000000e+00, 5.43709494e+02],  # [fx,  0, cx]
        [0.00000000e+00, 1.54981553e+03, 9.63609549e+02],  # [ 0, fy, cy]
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]   # [ 0,  0,  1]
    ])
    
    dist_coeffs = np.array([
        8.79441447e-02,   # k1: radial distortion
        1.63043039e-01,   # k2: radial distortion
        9.67939127e-03,   # p1: tangential distortion
        9.01387037e-04,   # p2: tangential distortion
        -9.22214736e-01   # k3: radial distortion
    ])
    
    return camera_matrix, dist_coeffs


def load_and_preprocess_images_square(image_path_list, target_size=1024, undistort_images=False):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation and alpha masks if present.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 1024.
        undistort_images (bool, optional): Whether to undistort images using known camera intrinsics. Defaults to False.

    Returns:
        tuple: When undistort_images=False: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 6) containing [x1, y1, x2, y2, width, height] for each image,
            torch.Tensor or None: Alpha masks with shape (N, 1, target_size, target_size) if present, None otherwise
        )
        tuple: When undistort_images=True: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 6) containing [x1, y1, x2, y2, width, height] for each image,
            numpy.ndarray: Updated camera matrix (3, 3) after optimal undistortion,
            tuple: ROI coordinates (x, y, w, h) used for cropping,
            torch.Tensor or None: Alpha masks with shape (N, 1, target_size, target_size) if present, None otherwise
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Get camera calibration constants
    camera_matrix, dist_coeffs = get_camera_intrinsics_for_undistortion()
    
    # Pre-calculate optimal camera matrix and ROI (same for all images)
    if undistort_images:
        # Assuming all images have the same shape, use first image to get dimensions
        sample_img = Image.open(image_path_list[0])
        sample_width, sample_height = sample_img.size
        
        # Adjust principal point for OpenCV undistortion (top-left corner convention)
        camera_matrix[0, 2] -= 0.5  # cx
        camera_matrix[1, 2] -= 0.5  # cy
        
        # Calculate optimal new camera matrix and ROI
        newK, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, 
            (sample_width, sample_height), 0  # alpha=0 for maximum valid pixels
        )        
        x, y, w, h = roi
        
        # Calculate undistortion maps
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, newK, 
            (sample_width, sample_height), cv2.CV_16SC2
        )
        
        print(f"Optimal undistortion ROI: ({x}, {y}, {w}, {h})")
        print(f"New camera matrix:\n{newK}")

    images = []
    masks = []
    has_alpha = False
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image with PIL for consistency with the rest of the pipeline
        img = Image.open(image_path)

        # Handle alpha channel
        alpha_mask = None
        if img.mode == "RGBA":
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            alpha_mask = a
            has_alpha = True
        else:
            img = img.convert("RGB")

        # Convert PIL image to numpy array for undistortion if needed
        if undistort_images:
            # Convert PIL to numpy array (OpenCV format: BGR)
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Apply undistortion using pre-computed maps
            img_array = cv2.remap(img_array, map1, map2, interpolation=cv2.INTER_CUBIC)
            # Crop to ROI (consistent across all images)
            img_array = img_array[y:y+h, x:x+w]
            
            # Convert back to PIL (RGB format)
            img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

            # Process alpha mask if present
            if alpha_mask is not None:
                # Apply undistortion to alpha mask using nearest neighbor to preserve binary values
                alpha_array = cv2.remap(np.array(alpha_mask), map1, map2, interpolation=cv2.INTER_NEAREST)
                # Crop to ROI
                alpha_array = alpha_array[y:y+h, x:x+w]
                alpha_mask = Image.fromarray(alpha_array)

        # Get original dimensions (after undistortion if applied)
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

        # Process alpha mask if present
        if alpha_mask is not None:
            # Create square alpha mask with same padding
            square_alpha = Image.new("L", (max_dim, max_dim), 0)  # Use 0 (transparent) for padding
            square_alpha.paste(alpha_mask, (left, top))
            
            # Resize to target size using nearest neighbor to preserve binary values
            square_alpha = square_alpha.resize((target_size, target_size), Image.Resampling.NEAREST)
            
            # Convert to tensor preserving original [0, 255] range
            alpha_tensor = torch.from_numpy(np.array(square_alpha)).unsqueeze(0)
            masks.append(alpha_tensor)
        else:
            # Create a fully opaque mask for images without alpha
            alpha_tensor = torch.ones((1, target_size, target_size)) * 255
            masks.append(alpha_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()
    
    # Stack alpha masks if any images had alpha channels
    masks_tensor = None
    if has_alpha:
        masks_tensor = torch.stack(masks)

    # Update camera matrix after all processing is complete
    if undistort_images:
        # Update the principal point based on our cropped region of interest (ROI)
        newK[0, 2] -= x
        newK[1, 2] -= y
        
        # Restore pixel center convention (add back the 0.5 we subtracted earlier)
        newK[0, 2] += 0.5
        newK[1, 2] += 0.5

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)
        if masks_tensor is not None and masks_tensor.dim() == 3:
            masks_tensor = masks_tensor.unsqueeze(0)

    # Return updated camera matrix if undistortion was applied
    if undistort_images:
        return images, original_coords, newK, roi, masks_tensor
    else:
        return images, original_coords, masks_tensor


# TODO: Implement mask support
def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        if img.mode == "RGBA":
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
        else:
            img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


# TODO: Implement mask support
def load_and_preprocess_images_no_resize(image_path_list, undistort_images=False):
    """
    Load images at their original resolution without any resizing or preprocessing.
    
    Args:
        image_path_list (list): List of paths to image files
        undistort_images (bool, optional): Whether to undistort images using known camera intrinsics. Defaults to False.
        
    Returns:
        tuple: When undistort_images=False: (
            torch.Tensor: Batched tensor of images with shape (N, 3, H, W),
            torch.Tensor: Array of shape (N, 6) containing [0, 0, W, H, W, H] for each image
        )
        tuple: When undistort_images=True: (
            torch.Tensor: Batched tensor of images with shape (N, 3, H, W),
            torch.Tensor: Array of shape (N, 6) containing [0, 0, W, H, W, H] for each image,
            numpy.ndarray: Updated camera matrix (3, 3) after optimal undistortion,
            tuple: ROI coordinates (x, y, w, h) used for cropping
        )
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Get camera calibration constants
    camera_matrix, dist_coeffs = get_camera_intrinsics_for_undistortion()
    
    # Pre-calculate optimal camera matrix and ROI if undistortion is enabled
    newK = None
    roi = None
    if undistort_images:
        # Use first image to get dimensions (assuming all images have same shape)
        sample_img = Image.open(image_path_list[0])
        sample_width, sample_height = sample_img.size
        
        # Calculate optimal new camera matrix and ROI
        newK, roi = get_optimal_camera_matrix_for_image_size(sample_width, sample_height)
        x, y, w, h = roi
        print(f"Optimal undistortion ROI: ({x}, {y}, {w}, {h})")
        print(f"New camera matrix:\n{newK}")
    
    images = []
    original_coords = []
    to_tensor = TF.ToTensor()
    
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)
        
        # Handle alpha channel
        if img.mode == "RGBA":
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
        else:
            img = img.convert("RGB")
        
        # Convert PIL image to numpy array for undistortion if needed
        if undistort_images:
            # Convert PIL to numpy array (OpenCV format: BGR)
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # Apply undistortion with optimal camera matrix
            img_array = cv2.undistort(img_array, camera_matrix, dist_coeffs, None, newK)
            # Crop to ROI (consistent across all images)
            img_array = img_array[y:y+h, x:x+w]
            # Convert back to PIL (RGB format)
            img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        
        original_width, original_height = img.size
        
        # Convert to tensor without any resizing
        img_tensor = to_tensor(img)
        images.append(img_tensor)
        
        # Store coordinates: [x1, y1, x2, y2, original_width, original_height]
        # Since no resizing, the image occupies the full tensor
        original_coords.append(np.array([0, 0, original_width, original_height, original_width, original_height]))
    
    # Stack images (they should all have the same size)
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()
    
    # Return updated camera matrix if undistortion was applied
    if undistort_images:
        return images, original_coords, newK, roi
    else:
        return images, original_coords