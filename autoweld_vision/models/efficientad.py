"""
Student-Teacher Feature Distillation Model (EfficientAD Paradigm) for AutoWeld-Vision.

Implements an unsupervised student-teacher anomaly detection model.
A frozen pre-trained "Teacher" network extracts semantic features from ImageNet,
and an active "Student" network is trained to replicate these features on normal data.
Anomalies are detected via local representation discrepancy (L2 feature difference).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any
from .base import BaseAnomalyModel
from .registry import ModelRegistry


class SlicedResNet(nn.Module):
    """Sliced ResNet-18 up to layer2 for extracting mid-level semantic features."""

    def __init__(self, pretrained: bool = True) -> None:
        """
        Initializes the sliced ResNet backbone.

        Args:
            pretrained: If True, loads pre-trained ImageNet weights.
        """
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting intermediate features."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class FeatureAutoEncoder(nn.Module):
    """AutoEncoder that reconstructs teacher intermediate features for anomaly profiling."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels // 4, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


@ModelRegistry.register("efficientad")
class EfficientADModel(BaseAnomalyModel):
    """
    Student-Teacher Feature Distillation & AutoEncoder Model.

    A frozen pre-trained Teacher network provides high-fidelity reference features.
    The Student network mimic the Teacher features, while an AutoEncoder reconstructs
    intermediate Teacher features for normal images.
    Anomalies trigger high student distillation gap and poor autoencoder reconstruction.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """Initializes the Student, Teacher, and AutoEncoder networks."""
        super().__init__()
        # Frozen Teacher network with pre-trained ImageNet weights
        self.teacher = SlicedResNet(pretrained=pretrained)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Active Student network with random initialization
        self.student = SlicedResNet(pretrained=False)

        # Active AutoEncoder branch
        self.autoencoder = FeatureAutoEncoder(channels=128)
        print(
            "Initialized Student-Teacher + AutoEncoder feature distillation model with ResNet-18 backbone."
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Runs model forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            A dictionary containing:
                - 'anomaly_map': The combined pixel-level anomaly maps.
                - 'score': The image-level maximum discrepancy scores.
        """
        self.teacher.eval()
        with torch.no_grad():
            teacher_features = self.teacher(x)

        student_features = self.student(x)
        reconstructed_features = self.autoencoder(teacher_features)

        # Compute L2 distance map channel-wise for distillation
        diff_distill = teacher_features - student_features
        map_distill = torch.mean(diff_distill**2, dim=1, keepdim=True)

        # Compute L2 distance map channel-wise for AutoEncoder reconstruction
        diff_ae = teacher_features - reconstructed_features
        map_reconstruct = torch.mean(diff_ae**2, dim=1, keepdim=True)

        # Fuse both maps (average of distillation and autoencoder reconstruction)
        anomaly_map = 0.5 * map_distill + 0.5 * map_reconstruct

        # Upsample anomaly map to original input dimensions
        anomaly_map = F.interpolate(
            anomaly_map,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Calculate image-level score as maximum of the discrepancy map
        batch_size = x.shape[0]
        score = anomaly_map.view(batch_size, -1).max(dim=1)[0].unsqueeze(1)

        return {
            "anomaly_map": anomaly_map,
            "score": score,
            "teacher_features": teacher_features,
            "student_features": student_features,
            "reconstructed_features": reconstructed_features,
        }
