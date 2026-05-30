"""
Student-Teacher Distillation Model (EfficientAD Concept) for AutoWeld-Vision.

Implements the core concept of EfficientAD (distillation discrepancy between
a frozen teacher and an active student) in a lightweight, high-performance local architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base import BaseAnomalyModel
from .registry import ModelRegistry


class FeatureExtractor(nn.Module):
    """Lightweight CNN Feature Extractor to serve as the backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        return x


@ModelRegistry.register("efficientad")
class EfficientADModel(BaseAnomalyModel):
    """
    Local Student-Teacher Distillation Model (EfficientAD Concept).

    A frozen 'Teacher' network extracts reference features, and an active
    'Student' network is trained to replicate these features on normal data.
    Anomaly maps are computed from the L2 feature discrepancy.
    """

    def __init__(self) -> None:
        """Initializes the Student and Teacher networks."""
        super().__init__()
        # Frozen Teacher network
        self.teacher = FeatureExtractor()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Active Student network
        self.student = FeatureExtractor()
        print("✓ Initialized SOTA-concept local Student-Teacher (EfficientAD) model.")

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Runs model forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            A dictionary containing:
                - 'anomaly_map': The pixel-level L2 discrepancy maps.
                - 'score': The image-level maximum discrepancy scores.
        """
        self.teacher.eval()
        with torch.no_grad():
            teacher_features = self.teacher(x)

        student_features = self.student(x)

        # Compute L2 distance map channel-wise
        diff = teacher_features - student_features
        anomaly_map = torch.mean(diff**2, dim=1, keepdim=True)

        # Upsample anomaly map to input dimensions
        anomaly_map = F.interpolate(
            anomaly_map,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Calculate image-level score as max of anomaly map
        batch_size = x.shape[0]
        score = anomaly_map.view(batch_size, -1).max(dim=1)[0].unsqueeze(1)

        return {"anomaly_map": anomaly_map, "score": score}
