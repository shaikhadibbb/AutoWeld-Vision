"""
PatchCore Model Wrapper for AutoWeld-Vision.

Integrates anomalib's memory-bank based PatchCore architecture with standard
type signatures and docstrings.
"""

from .base import BaseAnomalyModel
from .registry import ModelRegistry
import torch
from anomalib.models.image.patchcore import Patchcore
from typing import List, Dict, Any


@ModelRegistry.register("patchcore")
class PatchCoreModel(BaseAnomalyModel):
    """
    PatchCore anomaly detection model wrapper.

    Uses memory bank feature extraction with coreset subsampling for
    unsupervised industrial anomaly detection.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] = ["layer2", "layer3"],
    ) -> None:
        """
        Initializes the PatchCore model.

        Args:
            backbone: The CNN backbone architecture (default: 'wide_resnet50_2').
            layers: List of layers to extract features from (default: ['layer2', 'layer3']).
        """
        super().__init__()
        self.model = Patchcore(backbone=backbone)
        print(f"Initialized real PatchCore with {backbone}")

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Runs model forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            A dictionary containing:
                - 'anomaly_map': The pixel-level anomaly maps.
                - 'score': The image-level anomaly scores.
        """
        return self.model(x)
