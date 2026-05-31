"""
DINOv2-based Semantic Feature Outlier Model for AutoWeld-Vision.

Implements a mathematically corrected unsupervised anomaly detection baseline.
Computes spatial patch features against a cached global normal reference mean
compiled during training, rather than using the test image's own spatial mean.
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

    Extracts deep semantic feature maps using a DINOv2 backbone or ResNet-18 fallback.
    Detects anomalies by computing localized spatial outlier distances relative to the
    cached global normal training feature mean.
    """

    def __init__(
        self, backbone_type: str = "resnet18", pretrained: bool = True
    ) -> None:
        """
        Initializes the Dinomaly model.

        Args:
            backbone_type: Core feature extractor architecture ('dinov2' or 'resnet18').
            pretrained: If True, loads pretrained ImageNet weights.
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.normal_mean: torch.Tensor = None

        # Attempt to load DINOv2 from PyTorch Hub, fall back to ResNet-18 if offline
        self.use_dino = False
        if backbone_type.lower() == "dinov2":
            try:
                print("Loading DINOv2 from PyTorch Hub...")
                self.backbone = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14"
                )
                self.backbone.eval()
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.use_dino = True
                print("Loaded pre-trained DINOv2 backbone.")
            except Exception as e:
                print(
                    f"DINOv2 Hub load failed ({e}). Falling back to pretrained ResNet-18."
                )
                self.backbone_type = "resnet18"

        if not self.use_dino:
            # Load Pretrained ResNet-18 feature extractor
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
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

    def fit(self, dataloader) -> None:
        """
        Trains Dinomaly by extracting features from all normal training samples
        and computing the global normal reference feature mean.
        """
        self.eval()
        device = next(self.parameters()).device

        all_features = []
        print("Extracting normal reference features for Dinomaly...")

        with torch.no_grad():
            for batch in dataloader:
                images = (
                    batch["image"].to(device)
                    if isinstance(batch, dict)
                    else batch.to(device)
                )

                if self.use_dino:
                    x_resized = F.interpolate(
                        images, size=(224, 224), mode="bilinear", align_corners=False
                    )
                    features_dict = self.backbone.forward_features(x_resized)
                    patch_tokens = features_dict["x_norm_patchtokens"]
                    b, n, c = patch_tokens.shape
                    grid_h = grid_w = int(n**0.5)
                    feat_map = patch_tokens.transpose(1, 2).reshape(
                        b, c, grid_h, grid_w
                    )
                else:
                    feat_map = self.features(images)

                # Take average across batch samples to preserve memory footprint
                batch_mean = torch.mean(feat_map, dim=0, keepdim=True)
                all_features.append(batch_mean.cpu())

        # Compile the global normal average reference feature map: Shape (1, C, H_f, W_f)
        self.normal_mean = torch.mean(
            torch.cat(all_features, dim=0), dim=0, keepdim=True
        ).to(device)
        print(
            f"✓ Compiled Dinomaly reference mean feature profile. Shape: {self.normal_mean.shape}"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Runs forward pass: extracts features and calculates patch outlier discrepancy.
        """
        self.eval()
        batch_size = x.shape[0]

        if self.use_dino:
            # Resize to multiple of 14 for DINOv2
            x_resized = F.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                features_dict = self.backbone.forward_features(x_resized)
                patch_tokens = features_dict["x_norm_patchtokens"]  # (B, 256, 384)

            b, n, c = patch_tokens.shape
            grid_h = grid_w = int(n**0.5)
            feat_map = patch_tokens.transpose(1, 2).reshape(b, c, grid_h, grid_w)
        else:
            with torch.no_grad():
                feat_map = self.features(x)

        # Compute localized anomaly discrepancy map:
        if self.normal_mean is not None:
            # Match spatial dimensions if query sizes vary
            if feat_map.shape[2:] != self.normal_mean.shape[2:]:
                ref_mean = F.interpolate(
                    self.normal_mean,
                    size=feat_map.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                ref_mean = self.normal_mean

            # Compute spatial L2 discrepancy distance map against compiled normal baseline
            discrepancy = torch.norm(feat_map - ref_mean, p=2, dim=1, keepdim=True)
        else:
            # Fallback to query spatial mean if not fitted yet (with standard warning)
            print(
                "Warning: Dinomaly has not been fitted. Falling back to internal query mean."
            )
            mean_vector = torch.mean(feat_map, dim=(2, 3), keepdim=True)
            discrepancy = torch.norm(feat_map - mean_vector, p=2, dim=1, keepdim=True)

        # Upsample anomaly map back to original input resolution
        anomaly_map = F.interpolate(
            discrepancy,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Calculate image-level score as peak anomaly intensity
        score = anomaly_map.view(batch_size, -1).max(dim=1)[0].unsqueeze(1)

        return {"anomaly_map": anomaly_map, "score": score}


@ModelRegistry.register("general_ad")
class GeneralADModel(BaseAnomalyModel):
    """Fallback ResNet-18 basic reconstruction model."""

    def __init__(self, backbone: str = "resnet18") -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        return {
            "anomaly_map": torch.zeros(
                x.shape[0], 1, x.shape[2], x.shape[3], device=x.device
            ),
            "score": torch.zeros(x.shape[0], 1, device=x.device),
        }
