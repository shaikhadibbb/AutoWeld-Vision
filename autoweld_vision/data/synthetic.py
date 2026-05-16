import numpy as np
import cv2
import torch
from PIL import Image
import random
from typing import Tuple, List

class SyntheticAnomalyGenerator:
    """
    Tier C: Synthetic anomaly generation for welding defects.
    Implements CutPaste, Perlin Noise, and Blending logic.
    """
    
    @staticmethod
    def generate_perlin_noise_mask(shape: Tuple[int, int], scale: int = 10) -> np.ndarray:
        """Generates a fractal Perlin noise mask for defect simulation."""
        # Simplified Perlin-like noise using Gaussian Blur on random noise
        noise = np.random.normal(0, 1, (shape[0] // scale, shape[1] // scale))
        noise = cv2.resize(noise, shape, interpolation=cv2.INTER_LINEAR)
        noise = cv2.GaussianBlur(noise, (scale * 2 + 1, scale * 2 + 1), 0)
        mask = (noise > np.percentile(noise, 95)).astype(np.float32)
        return mask

    @staticmethod
    def apply_cutpaste(image: np.ndarray, patch_size_range: Tuple[int, int] = (10, 40)) -> np.ndarray:
        """Implements CutPaste augmentation: cuts a normal patch and pastes it elsewhere."""
        h, w, c = image.shape
        ps = random.randint(*patch_size_range)
        
        # Randomly select a patch
        from_y, from_x = random.randint(0, h-ps), random.randint(0, w-ps)
        patch = image[from_y:from_y+ps, from_x:from_x+ps].copy()
        
        # Color jitter the patch to make it anomalous
        patch = cv2.convertScaleAbs(patch, alpha=1.2, beta=10)
        
        # Paste it to a new location
        to_y, to_x = random.randint(0, h-ps), random.randint(0, w-ps)
        image[to_y:to_y+ps, to_x:to_x+ps] = patch
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[to_y:to_y+ps, to_x:to_x+ps] = 1
        return image, mask

    @staticmethod
    def simulate_porosity(image: np.ndarray, num_voids: int = 5) -> np.ndarray:
        """Simulates gas pockets (porosity) using dark elliptical voids."""
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        output = image.copy()
        
        for _ in range(num_voids):
            center = (random.randint(0, w), random.randint(0, h))
            axes = (random.randint(2, 6), random.randint(2, 6))
            cv2.ellipse(output, center, axes, 0, 0, 360, (20, 20, 20), -1)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            
        return output, mask / 255.0
