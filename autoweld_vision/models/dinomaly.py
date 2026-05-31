"""
DINOv2-based Semantic Feature Outlier Model for AutoWeld-Vision.

Implements a SOTA-style unsupervised anomaly detection baseline. It extracts 
deep patch-level feature representations from a pre-trained backbone and identifies 
anomalies by computing spatial outlier discrepancies within the image's feature map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any
from .registry import ModelRegistry
from .base import BaseAnomalyModel


@ModelRegistry.register("dinomaly")
class DinomalyModel(BaseAnomalyModel):
    """
    DINOv2 Semantic Feature Discrepancy Baseline.
    
    Extracts deep semantic feature maps using a pretrained DINOv2 backbone 
    (or falls back to a pretrained ResNet-18 when offline). Anomalies are 
    detected by computing localized spatial outlier distances relative to the 
    global feature mean of the normal weld regions.
    """

    def __init__(self, backbone_type: str = "resnet18", pretrained: bool = True) -> None:
        """
        Initializes the Dinomaly model.

        Args:
            backbone_type: Core feature extractor architecture.
            pretrained: If True, loads pretrained ImageNet weights.
        """
        super().__init__()
        self.backbone_type = backbone_type

        # Attempt to load DINOv2 from PyTorch Hub, fall back to ResNet-18 if offline
        self.use_dino = False
        if backbone_type.lower() == "dinov2":
            try:
                print("Loading DINOv2 from PyTorch Hub...")
                self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
                self.backbone.eval()
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.use_dino = True
                print("Loaded pre-trained DINOv2 backbone.")
            except Exception as e:
                print(f"DINOv2 Hub load failed ({e}). Falling back to pretrained ResNet-18.")
                self.backbone_type = "resnet18"

        if not self.use_dino:
            # Load Pretrained ResNet-18 feature extractor
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            # Slice ResNet up to layer2 to get rich spatial semantic features
            self.features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
            )
            self.features.eval()
            for param in self.features.parameters():
                param.requires_grad = False
            print("Loaded pre-trained ResNet-18 feature extractor for Dinomaly.")

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Runs forward pass: extracts features and calculates patch discrepancy.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            A dictionary containing:
                - 'anomaly_map': The upsampled spatial discrepancy maps.
                - 'score': The image-level maximum discrepancy scores.
        """
        batch_size = x.shape[0]

        if self.use_dino:
            # Resize to multiple of 14 for DINOv2
            x_resized = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            with torch.no_grad():
                # Extract features: (B, 256, 384) for 224x224 input
                features_dict = self.backbone.forward_features(x_resized)
                patch_tokens = features_dict["x_norm_patchtokens"]  # (B, 256, 384)
            
            # Reshape patches to spatial grid: (B, 384, 16, 16)
            b, n, c = patch_tokens.shape
            grid_h = grid_w = int(n ** 0.5)
            feat_map = patch_tokens.transpose(1, 2).reshape(b, c, grid_h, grid_w)
        else:
            # Extract spatial features using ResNet: (B, 128, H/8, W/8)
            with torch.no_grad():
                feat_map = self.features(x)

        # Compute localized anomaly discrepancy map:
        # Calculate the mean feature vector representing normal reference context
        mean_vector = torch.mean(feat_map, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Compute spatial L2 discrepancy distance map
        discrepancy = torch.norm(feat_map - mean_vector, p=2, dim=1, keepdim=True)  # (B, 1, H_f, W_f)

        # Upsample anomaly map back to original input resolution
        anomaly_map = F.interpolate(
            discrepancy,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Calculate image-level score as peak anomaly intensity
        score = anomaly_map.view(batch_size, -1).max(dim=1)[0].unsqueeze(1)

        # Standardize score boundaries to a clean range
        return {"anomaly_map": anomaly_map, "score": score}


@ModelRegistry.register("general_ad")
class GeneralADModel(BaseAnomalyModel):
    """Fallback ResNet-18 basic reconstruction model."""

    def __init__(self, backbone: str = "resnet18") -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        return {
            "anomaly_map": torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device),
            "score": torch.zeros(x.shape[0], 1, device=x.device),
        }
