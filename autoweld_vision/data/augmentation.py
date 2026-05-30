"""
CutPaste Data Augmentation for Industrial Anomaly Detection.

This module implements the CutPaste augmentation pipeline, which cuts a random
patch from a normal image, applies color and geometric perturbations, and
pastes it back to create a synthetic anomaly with a pixel-level ground-truth mask.
"""

import random
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms as T
from typing import Tuple, Dict, Any

class CutPaste:
    """
    CutPaste data augmentation for self-supervised anomaly detection.
    
    Extracts a random patch from a normal image, transforms it (color jitter, 
    rotation, etc.), and pastes it back onto a random location on the image.
    Generates both the augmented image and a ground-truth anomaly mask.
    """
    
    def __init__(
        self,
        patch_ratio_bounds: Tuple[float, float] = (0.02, 0.16),
        aspect_ratio_bounds: Tuple[float, float] = (0.3, 3.3),
        color_jitter: bool = True,
        rotation_angle_bounds: Tuple[float, float] = (-45.0, 45.0)
    ) -> None:
        """
        Initializes the CutPaste augmentation.
        
        Args:
            patch_ratio_bounds: Min and max area ratio of the patch relative to the image.
            aspect_ratio_bounds: Min and max aspect ratio of the patch.
            color_jitter: If True, applies color jitter to the cut patch.
            rotation_angle_bounds: Range of angles (in degrees) to rotate the patch.
        """
        self.patch_ratio_bounds = patch_ratio_bounds
        self.aspect_ratio_bounds = aspect_ratio_bounds
        self.rotation_angle_bounds = rotation_angle_bounds
        
        if color_jitter:
            self.jitter = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        else:
            self.jitter = None

    def __call__(self, img: Image.Image) -> Dict[str, Any]:
        """
        Applies CutPaste augmentation on the input PIL Image.
        
        Args:
            img: A PIL Image of the original normal image.
            
        Returns:
            A dictionary containing:
                - 'image': The augmented PIL Image with the synthetic anomaly.
                - 'mask': The binary PIL Image ground-truth mask (255 for anomaly, 0 for normal).
                - 'box': The bounding box coordinates (xmin, ymin, xmax, ymax) of the patch.
        """
        w, h = img.size
        img_area = w * h
        
        # 1. Determine patch dimensions
        patch_area = random.uniform(*self.patch_ratio_bounds) * img_area
        aspect_ratio = random.uniform(*self.aspect_ratio_bounds)
        
        # Calculate patch height and width
        patch_h = int(np.sqrt(patch_area * aspect_ratio))
        patch_w = int(np.sqrt(patch_area / aspect_ratio))
        
        # Clamp patch dimensions to image size
        patch_h = max(2, min(patch_h, h - 2))
        patch_w = max(2, min(patch_w, w - 2))
        
        # 2. Select random source location to cut the patch
        from_x = random.randint(0, w - patch_w)
        from_y = random.randint(0, h - patch_h)
        
        # Crop the patch
        patch = img.crop((from_x, from_y, from_x + patch_w, from_y + patch_h))
        
        # 3. Apply transformations to the patch
        if self.jitter is not None:
            patch = self.jitter(patch)
            
        # Optional rotation
        angle = random.uniform(*self.rotation_angle_bounds)
        # Convert patch to RGBA to rotate without black borders
        patch_rgba = patch.convert("RGBA")
        rotated_rgba = patch_rgba.rotate(angle, expand=True, resample=Image.BICUBIC)
        # Create a mask for pasting based on alpha channel
        patch_mask = rotated_rgba.split()[3]
        patch = rotated_rgba.convert("RGB")
        
        # Get new patch dimensions after rotation expansion
        p_w, p_h = patch.size
        
        # 4. Select random destination location to paste
        to_x = random.randint(0, w - p_w) if w > p_w else 0
        to_y = random.randint(0, h - p_h) if h > p_h else 0
        
        # 5. Paste patch and create mask
        augmented_img = img.copy()
        augmented_img.paste(patch, (to_x, to_y), patch_mask)
        
        # Generate binary ground truth mask
        mask = Image.new("L", (w, h), 0)
        mask.paste(255, (to_x, to_y), patch_mask)
        
        return {
            "image": augmented_img,
            "mask": mask,
            "box": (to_x, to_y, to_x + p_w, to_y + p_h)
        }
