"""
Unit tests for the CutPaste data augmentation pipeline.
"""

import pytest
import numpy as np
from PIL import Image
from autoweld_vision.data.augmentation import CutPaste

def test_cutpaste_augmentation_output() -> None:
    """Verifies that CutPaste produces expected images, masks, and boundary bounds."""
    # Create a simple solid white image (256x256)
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    
    augmentor = CutPaste(
        patch_ratio_bounds=(0.05, 0.1),
        aspect_ratio_bounds=(0.5, 2.0),
        color_jitter=True
    )
    
    result = augmentor(img)
    
    # 1. Assert structure
    assert "image" in result
    assert "mask" in result
    assert "box" in result
    
    # 2. Assert types and sizes
    assert isinstance(result["image"], Image.Image)
    assert isinstance(result["mask"], Image.Image)
    assert result["image"].size == (256, 256)
    assert result["mask"].size == (256, 256)
    
    # 3. Assert mask contains values other than 0 (i.e. defect region is present)
    mask_np = np.array(result["mask"])
    assert np.any(mask_np == 255)
    assert np.all((mask_np == 0) | (mask_np == 255))
    
    # 4. Assert bounding box bounds
    xmin, ymin, xmax, ymax = result["box"]
    assert 0 <= xmin < xmax <= 256
    assert 0 <= ymin < ymax <= 256
