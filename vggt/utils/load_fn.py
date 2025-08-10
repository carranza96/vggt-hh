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


def get_camera_intrinsics(undistorted_images=True):
    """
    Get the original camera matrix and distortion coefficients.
    
    Returns:
        tuple: (camera_matrix, dist_coeffs) as numpy arrays
    """
    if undistorted_images:
        camera_matrix = np.array([
            [1.57984692e+03, 0.00000000e+00, 5.44577814e+02], # [fx,  0, cx]
            [0.00000000e+00, 1.55774009e+03, 9.84074151e+02], # [ 0, fy, cy]
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]   # [ 0,  0,  1]
        ])
        dist_coeffs = np.zeros(5)   
    else:
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
        tuple: Always returns (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 6) containing [x1, y1, x2, y2, width, height] for each image,
            numpy.ndarray: Camera matrix (3, 3) - adjusted for padding transformation,
            tuple: ROI coordinates (x, y, w, h) - optimal ROI if undistorted, (0, 0, width, height) if not,
            torch.Tensor or None: Alpha masks with shape (N, 1, target_size, target_size) if present, None otherwise
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Get camera calibration constants
    camera_matrix, dist_coeffs = get_camera_intrinsics(undistorted_images=not undistort_images)

    # Assuming all images have the same shape, use first image to get dimensions
    sample_img = Image.open(image_path_list[0])
    sample_width, sample_height = sample_img.size
    print(f"Original camera matrix for images {sample_height}x{sample_width}")
    print(f"fx={camera_matrix[0, 0]}; fy={camera_matrix[1, 1]}; cx={camera_matrix[0, 2]}; cy={camera_matrix[1, 2]}")
    print(f"Original distortion coefficients: {dist_coeffs}")
    
    # Initialize return values
    newK = camera_matrix.copy()  # Default to original camera matrix
    roi = None  # Will be set later based on undistortion or original image size
    
    # Pre-calculate optimal camera matrix and ROI (same for all images)
    if undistort_images:     
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
            (sample_width, sample_height), cv2.CV_32FC1
        )
        
        # Apply undistortion-related camera matrix adjustments
        # Update the principal point based on our cropped region of interest (ROI)
        newK[0, 2] -= x  # cx -= x
        newK[1, 2] -= y  # cy -= y
        
        # Restore pixel center convention (add back the 0.5 we subtracted earlier)
        newK[0, 2] += 0.5
        newK[1, 2] += 0.5
        
        print(f"Undistortion ROI: ({x}, {y}, {w}, {h})")
        print(f"Camera matrix after undistortion adjustments:")
        print(f"fx={newK[0, 0]}; fy={newK[1, 1]}; cx={newK[0, 2]}; cy={newK[1, 2]}")
    else:
        # For non-undistorted images, we'll set ROI based on first image dimensions
        sample_img = Image.open(image_path_list[0])
        sample_width, sample_height = sample_img.size
        roi = (0, 0, sample_width, sample_height)

    images = []
    masks = []
    has_alpha = False
    original_coords = []
    to_tensor = TF.ToTensor()
    pil_imgs = []
    pil_masks = []

    # First pass: load images and masks, store PIL objects for cropping
    for image_path in image_path_list:
        img = Image.open(image_path)
        alpha_mask = None
        if img.mode == "RGBA":
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            alpha_mask = a
            has_alpha = True
        else:
            img = img.convert("RGB")
        pil_imgs.append(img)
        pil_masks.append(alpha_mask)

    # If any mask is present, compute min_y, max_y, min_x, max_x across all masks
    min_y, max_y, min_x, max_x = None, None, None, None
    if any(m is not None for m in pil_masks):
        mask_arrays = [np.array(m) if m is not None else None for m in pil_masks]
        y_indices = []
        x_indices = []
        for arr in mask_arrays:
            if arr is not None:
                ys, xs = np.where(arr > 0)
                if len(ys) > 0:
                    y_indices.append(ys)
                if len(xs) > 0:
                    x_indices.append(xs)
        if y_indices:
            min_y = min([ys.min() for ys in y_indices])
            max_y = max([ys.max() for ys in y_indices])
        if x_indices:
            min_x = min([xs.min() for xs in x_indices])
            max_x = max([xs.max() for xs in x_indices])

    # Second pass: process images and masks, apply vertical and horizontal crop if needed
    for idx, (img, alpha_mask) in enumerate(zip(pil_imgs, pil_masks)):
        # Convert PIL image to numpy array for undistortion if needed
        if undistort_images:
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_array = cv2.remap(img_array, map1, map2, interpolation=cv2.INTER_CUBIC)
            img_array = img_array[y:y+h, x:x+w]
            img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            if alpha_mask is not None:
                alpha_array = cv2.remap(np.array(alpha_mask), map1, map2, interpolation=cv2.INTER_NEAREST)
                alpha_array = alpha_array[y:y+h, x:x+w]
                alpha_mask = Image.fromarray(alpha_array)

        # Apply vertical crop if min_y and max_y are set
        crop_top, crop_bottom = 0, None
        crop_left, crop_right = 0, None
        width, height = img.size
        if min_y is not None and max_y is not None:
            crop_top = min_y
            crop_bottom = max_y + 1  # +1 to include max_y
        else:
            crop_bottom = height
        if min_x is not None and max_x is not None:
            crop_left = min_x
            crop_right = max_x + 1  # +1 to include max_x
        else:
            crop_right = width
        # Crop image if needed
        if (crop_top != 0 or crop_bottom != height) or (crop_left != 0 or crop_right != width):
            img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            if alpha_mask is not None:
                alpha_mask = alpha_mask.crop((crop_left, crop_top, crop_right, crop_bottom))

        width, height = img.size
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        scale = target_size / max_dim
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)
        if alpha_mask is not None:
            square_alpha = Image.new("L", (max_dim, max_dim), 0)
            square_alpha.paste(alpha_mask, (left, top))
            square_alpha = square_alpha.resize((target_size, target_size), Image.Resampling.NEAREST)
            alpha_tensor = torch.from_numpy(np.array(square_alpha)).unsqueeze(0)
            masks.append(alpha_tensor)
        else:
            alpha_tensor = torch.ones((1, target_size, target_size)) * 255
            masks.append(alpha_tensor)

    # Adjust camera matrix accounting for cropping, padding, and resizing
    # CROPPING
    # Rectify intrinsics due to cropping: principal point cy -= crop_top, cx -= crop_left
    newK[1, 2] -= crop_top
    newK[0, 2] -= crop_left
    print("Global crop using masks")
    print(f"Crop top:{crop_top}; Crop bottom:{crop_bottom};")
    print(f"Crop left:{crop_left}; Crop right:{crop_right};")
    print(f"New camera matrix after cropping (cx-crop_left; cy-crop_top):")
    print(f"fx={newK[0, 0]}; fy={newK[1, 1]}; cx={newK[0, 2]}; cy={newK[1, 2]};")

    # Get original dimensions from first image's coordinates
    original_width = int(original_coords[0][4].item())
    original_height = int(original_coords[0][5].item())
    
    print(f"Image size after cropping: {original_height}x{original_width}")
    print(f"Target square size: {target_size}x{target_size}")
    
    # PADDING
    # Calculate padding transformation
    if original_width < original_height:
        padding_x = (original_height - original_width) / 2
        padding_y = 0
    else:
        padding_x = 0
        padding_y = (original_width - original_height) / 2
    
    print(f"Padding: x={padding_x}, y={padding_y}")
    
    # Apply padding transformation to camera matrix
    newK[0, 2] += padding_x  # cx += padding_x
    newK[1, 2] += padding_y  # cy += padding_y
    
    print(f"Camera matrix after padding adjustment:")
    print(f"fx={newK[0, 0]}; fy={newK[1, 1]}; cx={newK[0, 2]}; cy={newK[1, 2]}")
    
    # RESIZING
    # Apply scale factor adjustment after padding
    max_dim = max(original_width, original_height)
    scale_factor = target_size / max_dim
    print(f"Scale factor: {scale_factor}")
    
    newK[:2, :] *= scale_factor  # Scale fx, fy, cx, cy
    
    print(f"Final camera matrix after scale adjustment:")
    print(f"fx={newK[0, 0]}; fy={newK[1, 1]}; cx={newK[0, 2]}; cy={newK[1, 2]}")

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()
    
    # Stack alpha masks if any images had alpha channels
    masks_tensor = None
    if has_alpha:
        masks_tensor = torch.stack(masks)
        
    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)
        if masks_tensor is not None and masks_tensor.dim() == 3:
            masks_tensor = masks_tensor.unsqueeze(0)

    return images, original_coords, newK, roi, masks_tensor


# TODO: Adapt to new code
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

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
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
