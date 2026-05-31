"""
Late Fusion Ensemble and Defect Gating for Industrial Anomaly Detection.

This module implements AnomalyEnsemble which fuses the anomaly scores and anomaly maps
from PatchCore and Student-Teacher models. The score fusion weights are learned 
programmatically by minimizing binary cross-entropy validation loss on validation splits.
Includes DefectRouter which routes defects using spatial geometry.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from scipy.optimize import minimize
from .base import BaseAnomalyModel


class AnomalyNormalizer:
    """
    Fits and normalizes anomaly scores to [0, 1] using min-max scaling
    learned from validation/calibration sets.
    """
    def __init__(self) -> None:
        self.min_val: float = 0.0
        self.max_val: float = 1.0
        self.is_fitted: bool = False

    def fit(self, scores: np.ndarray) -> None:
        """Fits min and max values on validation scores."""
        if len(scores) == 0:
            return
        self.min_val = float(np.min(scores))
        self.max_val = float(np.max(scores))
        if self.max_val == self.min_val:
            self.max_val += 1e-8
        self.is_fitted = True

    def normalize(self, scores: Any) -> Any:
        """Normalizes scores (numpy array or torch tensor) to [0, 1]."""
        if isinstance(scores, torch.Tensor):
            return torch.clamp(
                (scores - self.min_val) / (self.max_val - self.min_val + 1e-8),
                0.0,
                1.0,
            )
        else:
            return np.clip(
                (scores - self.min_val) / (self.max_val - self.min_val + 1e-8),
                0.0,
                1.0,
            )


class AnomalyEnsemble(BaseAnomalyModel):
    """
    Weighted Late Fusion Ensemble for Anomaly Detection.

    Combines anomaly scores and anomaly maps from multiple models (e.g. PatchCore and EfficientAD).
    The weights are learned dynamically by minimizing validation binary cross-entropy loss.
    """

    def __init__(self, models: Dict[str, nn.Module]) -> None:
        """
        Initializes the AnomalyEnsemble.

        Args:
            models: A dictionary mapping model names to their instantiated nn.Module classes.
        """
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.w1: float = 0.5
        self.w2: float = 0.5
        self._learned_weights: Dict[str, float] = {
            name: 1.0 / len(models) for name in models.keys()
        }
        self.normalizers = {name: AnomalyNormalizer() for name in models.keys()}

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Runs forward inference across all models in the ensemble and fuses results.

        Args:
            x: Input torch.Tensor.

        Returns:
            A dictionary containing:
                - 'score': The final fused anomaly score (float tensor).
                - 'anomaly_map': The fused pixel-level anomaly map.
                - 'model_scores': Dictionary of individual model scores.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name, model in self.models.items():
            results[name] = model(x)

        # Extract individual scores
        scores: Dict[str, torch.Tensor] = {
            name: (res.get("pred_score", res.get("score")) if isinstance(res, dict) else res)
            for name, res in results.items()
        }

        # Weighted fusion of scores
        fused_score = torch.zeros_like(list(scores.values())[0])
        for name, score in scores.items():
            norm_score = self.normalizers[name].normalize(score)
            fused_score += self._learned_weights[name] * norm_score

        # Weighted fusion of anomaly maps
        fused_map = torch.zeros_like(list(results.values())[0]["anomaly_map"])
        for name, res in results.items():
            norm_map = self.normalizers[name].normalize(res["anomaly_map"])
            fused_map += self._learned_weights[name] * norm_map

        return {"score": fused_score, "anomaly_map": fused_map, "model_scores": scores}

    def optimize_weights(
        self,
        val_scores_m1: np.ndarray,
        val_scores_m2: np.ndarray,
        val_labels: np.ndarray,
    ) -> None:
        """
        Optimizes score fusion weights by minimizing Binary Cross Entropy loss on a validation set.

        Uses scipy.optimize.minimize to find weights w1, w2 that minimize BCE.

        Args:
            val_scores_m1: Array of validation anomaly scores from model 1.
            val_scores_m2: Array of validation anomaly scores from model 2.
            val_labels: Binary ground-truth labels (0 for normal, 1 for anomaly).
        """
        model_names = list(self.models.keys())
        if len(model_names) >= 2:
            self.normalizers[model_names[0]].fit(val_scores_m1)
            self.normalizers[model_names[1]].fit(val_scores_m2)
            
            norm_val_m1 = self.normalizers[model_names[0]].normalize(val_scores_m1)
            norm_val_m2 = self.normalizers[model_names[1]].normalize(val_scores_m2)
        else:
            norm_val_m1 = val_scores_m1
            norm_val_m2 = val_scores_m2

        def bce_loss(weights: np.ndarray) -> float:
            w1, w2 = weights
            fused = w1 * norm_val_m1 + w2 * norm_val_m2
            # Clip probabilities to avoid log(0)
            probs = np.clip(fused, 1e-7, 1.0 - 1e-7)
            loss = -np.mean(
                val_labels * np.log(probs) + (1.0 - val_labels) * np.log(1.0 - probs)
            )
            return float(loss)

        # Constraints: w1 + w2 = 1 and 0 <= w1, w2 <= 1
        cons = {"type": "eq", "fun": lambda w: w[0] + w[1] - 1.0}
        bounds = ((0.0, 1.0), (0.0, 1.0))

        initial_weights = np.array([0.5, 0.5])
        res = minimize(
            bce_loss, initial_weights, method="SLSQP", bounds=bounds, constraints=cons
        )

        if res.success:
            w1, w2 = res.x
            self.w1 = float(w1)
            self.w2 = float(w2)
            if len(model_names) >= 2:
                self._learned_weights[model_names[0]] = self.w1
                self._learned_weights[model_names[1]] = self.w2
            print(
                f"✓ Ensemble weights optimized! w1 ({model_names[0]}): {self.w1:.4f}, w2 ({model_names[1]}): {self.w2:.4f}"
            )
        else:
            print("⚠️ Weight optimization failed. Reverting to uniform weights.")


class DefectRouter:
    """
    Defect-Type-Specific Routing Gating Mechanism.

    Uses a deterministic geometric shape and area profiling classifier to route 
    defects to specialized downstream models based on structural morphology.
    """

    def __init__(self, specialists: Dict[str, nn.Module]) -> None:
        """
        Initializes the DefectRouter.

        Args:
            specialists: A dictionary mapping defect categories to specialized models.
        """
        self.specialists = specialists

    def route(self, image: torch.Tensor, coarse_map: torch.Tensor) -> str:
        """
        Routes the image to the appropriate specialist model based on spatial geometry.

        Args:
            image: The input image tensor (B, C, H, W).
            coarse_map: The initial coarse anomaly map tensor (B, 1, H, W).

        Returns:
            The name of the chosen specialist defect model.
        """
        # Threshold the coarse anomaly map to isolate defect regions
        mask = (coarse_map > 0.5).float()

        # Compute spatial footprint size
        defect_area = float(mask.sum().item())

        # If no significant defect is detected, route to standard base specialist
        if defect_area < 50:
            return list(self.specialists.keys())[0]

        # Calculate defect spatial span along height and width
        proj_h = mask.sum(dim=(0, 1, 3)) > 0
        proj_w = mask.sum(dim=(0, 1, 2)) > 0

        indices_h = torch.where(proj_h)[0]
        indices_w = torch.where(proj_w)[0]

        if len(indices_h) == 0 or len(indices_w) == 0:
            return list(self.specialists.keys())[0]

        span_h = float(indices_h[-1] - indices_h[0] + 1)
        span_w = float(indices_w[-1] - indices_w[0] + 1)

        aspect_ratio = span_h / (span_w + 1e-6)

        # Geometric routing rules:
        # 1. Elongated structures (cracks, lack of fusion) route to Crack/Cable Specialists.
        # 2. Localized circular shapes (porosity, pores) route to Pore/Bottle Specialists.
        # 3. High area complex defects route to primary base ensembled model.
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            for key in self.specialists.keys():
                if "crack" in key.lower() or "cable" in key.lower():
                    return key
        elif defect_area < 1200:
            for key in self.specialists.keys():
                if "pore" in key.lower() or "bottle" in key.lower():
                    return key

        # Fallback to the first registered specialist
        return list(self.specialists.keys())[0]
