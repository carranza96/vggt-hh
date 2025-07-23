"""
Utility functions and classes for VGGT handheld reconstruction
"""

import os
import sys
import numpy as np
import torch


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


class GPUMemoryMonitor:
    """GPU Memory monitoring utilities"""
    
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


class Tee:
    """Simple tee class to write to both console and file"""
    
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
