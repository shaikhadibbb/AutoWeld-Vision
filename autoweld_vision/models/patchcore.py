"""
PatchCore Model Implementation in Pure PyTorch for AutoWeld-Vision.

Implements the complete PatchCore anomaly detection paradigm (Roth et al., 2022)
from scratch in pure PyTorch, removing all dependencies on external anomaly
frameworks like Anomalib.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict, Any
from .base import BaseAnomalyModel
from .registry import ModelRegistry


@ModelRegistry.register("patchcore")
class PatchCoreModel(BaseAnomalyModel):
    """
    Scratch-built PyTorch PatchCore Anomaly Detection Model.
    
    Extracts intermediate features from a pre-trained backbone, aggregate neighborhood
    context using local pooling, compiles a coreset memory bank via greedy subsampling,
    and detects anomalies using scaled K-nearest neighbor distances.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] = ["layer2", "layer3"],
        coreset_sampling_ratio: float = 0.1,
    ) -> None:
        """
        Initializes the scratch PatchCore model.

        Args:
            backbone: Pretrained feature extractor backbone ('wide_resnet50_2' or 'resnet18').
            layers: Model layers to hook features from.
            coreset_sampling_ratio: Fraction of training patches to preserve in coreset.
        """
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        self.coreset_sampling_ratio = coreset_sampling_ratio

        # Load backbone with weights
        if backbone == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            self.feature_extractor = models.wide_resnet50_2(weights=weights)
        else:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.feature_extractor = models.resnet18(weights=weights)

        # Freeze all backbone layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        # Hook registration structures
        self.hooks = []
        self.features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

        # Register memory bank as a buffer for automatic state_dict serialization
        self.register_buffer("memory_bank", torch.empty(0))
        print(f"Initialized scratch-built PatchCore with {backbone} backbone.")

    def _register_hooks(self) -> None:
        """Registers forward hooks on target intermediate layer activations."""
        def get_activation(name: str):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        for name in self.layers:
            layer = getattr(self.feature_extractor, name)
            h = layer.register_forward_hook(get_activation(name))
            self.hooks.append(h)

    def _extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """
        Runs backbone, extracts target layer activations, applies local aggregation
        pooling, and reshapes patches into a unified channel representation.
        """
        self.features.clear()
        _ = self.feature_extractor(x)

        pooled_features = []
        target_size = None

        for name in self.layers:
            feat = self.features[name]
            # Aggregates spatial context: avg pooling over size 3, stride 1, padding 1
            pooled_feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)

            if target_size is None:
                target_size = pooled_feat.shape[2:]

            if pooled_feat.shape[2:] != target_size:
                pooled_feat = F.interpolate(
                    pooled_feat, size=target_size, mode="bilinear", align_corners=False
                )
            pooled_features.append(pooled_feat)

        # Concatenate along channel dimension: Shape (B, C_total, H_f, W_f)
        concatenated = torch.cat(pooled_features, dim=1)

        # Permute and reshape to (B * H_f * W_f, C_total)
        b, c, h, w = concatenated.shape
        patches = concatenated.permute(0, 2, 3, 1).reshape(-1, c)
        return patches, (b, h, w)

    def fit(self, dataloader) -> None:
        """
        Trains PatchCore: extracts patches from normal data, runs greedy coreset
        selection to build memory bank, and saves final coreset tensor.
        """
        self.feature_extractor.eval()
        device = next(self.feature_extractor.parameters()).device

        all_patches = []
        print("Extracting training patches from normal split...")

        with torch.no_grad():
            for batch in dataloader:
                # Dataloader can return dict or raw tensor
                images = batch["image"].to(device) if isinstance(batch, dict) else batch.to(device)
                patches, _ = self._extract_features(images)
                all_patches.append(patches.cpu())

        # Compile memory bank patches: shape (N_patches, C_total)
        memory_bank_raw = torch.cat(all_patches, dim=0)
        print(f"Memory bank patches gathered: {memory_bank_raw.shape[0]}. Running greedy coreset selection...")

        # Subsample to a highly representative subset (capped at 1500 elements for speed)
        coreset = self._greedy_coreset_selection(memory_bank_raw, self.coreset_sampling_ratio)
        self.memory_bank = coreset.to(device)
        print(f"✓ Coreset memory bank compiled successfully! Selected patches: {self.memory_bank.shape[0]}")

    def _greedy_coreset_selection(self, patches: torch.Tensor, ratio: float) -> torch.Tensor:
        """Greedy coreset selection selects representative vectors using minimax distance."""
        num_patches = patches.shape[0]
        n_select = max(10, int(num_patches * ratio))
        # Keep selected size capped at 1500 for runtime latency efficiency
        n_select = min(n_select, 1500)

        selected_indices = []
        # Seed with a random patch vector
        start_idx = torch.randint(0, num_patches, (1,)).item()
        selected_indices.append(start_idx)

        # Initialize distance tracking
        selected_patch = patches[start_idx].unsqueeze(0)
        min_distances = torch.norm(patches - selected_patch, p=2, dim=1)

        # Iteratively select the most distant patches
        for _ in range(1, n_select):
            new_idx = torch.argmax(min_distances).item()
            selected_indices.append(new_idx)

            # Update min distances with the new addition
            new_patch = patches[new_idx].unsqueeze(0)
            new_distances = torch.norm(patches - new_patch, p=2, dim=1)
            min_distances = torch.min(min_distances, new_distances)

        return patches[selected_indices]

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Performs PatchCore forward pass: extracts features, computes L2 nearest neighbor
        distances against coreset, constructs upsampled anomaly map, and returns KNN-scaled score.
        """
        self.feature_extractor.eval()
        device = x.device

        if self.memory_bank is None or self.memory_bank.numel() == 0:
            # If model state loaded but memory_bank empty, load random/default mock or raise error
            raise RuntimeError("PatchCore coreset memory bank has not been fitted/trained yet!")

        # Extract features: patches has shape (B * H_f * W_f, C)
        patches, (b, h_f, w_f) = self._extract_features(x)

        # Compute distance matrix between query patches and coreset: shape (N_test, N_coreset)
        # Using expanded L2 norm formulation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b^T
        a_norm = torch.sum(patches**2, dim=1, keepdim=True)
        b_norm = torch.sum(self.memory_bank**2, dim=1, keepdim=True).t()
        dot_product = torch.matmul(patches, self.memory_bank.t())

        dist_matrix = torch.clamp(a_norm + b_norm - 2 * dot_product, min=0.0)
        dist_matrix = torch.sqrt(dist_matrix + 1e-8)  # shape (N_test, N_coreset)

        # Find nearest neighbor distance for each test patch
        min_dists, min_indices = torch.min(dist_matrix, dim=1)

        # Reshape to spatial activation grid: Shape (B, 1, H_f, W_f)
        anomaly_map = min_dists.reshape(b, h_f, w_f).unsqueeze(1)

        # Upsample anomaly map to input image resolution
        anomaly_map = F.interpolate(
            anomaly_map, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
        )

        # Compute image-level anomaly score using scaled KNN distance formulation
        max_dist, max_patch_idx = torch.max(min_dists, dim=0)

        # Find closest neighbors in coreset for max_patch
        worst_patch_dists = dist_matrix[max_patch_idx]
        sorted_worst_dists, _ = torch.sort(worst_patch_dists)
        k = 9
        k_nearest_dists = sorted_worst_dists[:k]

        # Calculate soft-max scale factor based on nearest neighbor density
        weights = torch.exp(-k_nearest_dists)
        scale_factor = 1.0 - (torch.exp(-max_dist) / (torch.sum(weights) + 1e-8))
        image_score = (max_dist * scale_factor).reshape(b, 1)

        return {
            "anomaly_map": anomaly_map,
            "score": image_score,
        }

    def __del__(self) -> None:
        """Cleans up target hooks on model deletion."""
        for hook in self.hooks:
            hook.remove()
