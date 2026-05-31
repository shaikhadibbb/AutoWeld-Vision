"""
Late Fusion Ensemble and Defect Gating for Industrial Anomaly Detection.

This module implements AnomalyEnsemble which fuses the anomaly scores and anomaly maps
from PatchCore and Student-Teacher models. The score fusion weights are learned
programmatically by minimizing binary cross-entropy loss on Platt-calibrated validation splits.
Includes DefectRouter which routes defects using spatial geometry.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union
import numpy as np
from scipy.optimize import minimize
from .base import BaseAnomalyModel


class PlattCalibrator:
    """
    Fits and applies Platt Scaling (Logistic Calibration) to map raw, uncalibrated
    distance-based anomaly scores into statistically sound probabilities in [0, 1].

    Fits the model P(y=1 | S) = 1 / (1 + exp(A * S + B)) using validation binary labels.
    """

    def __init__(self) -> None:
        self.A: float = -1.0
        self.B: float = 0.0
        self.is_fitted: bool = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fits calibration parameters A and B by maximizing Bernoulli log-likelihood."""
        if len(scores) == 0:
            return

        def neg_log_likelihood(params: np.ndarray) -> float:
            A, B = params
            logits = A * scores + B

            # Safe cross-entropy implementation to avoid numerical underflow/overflow
            loss = np.zeros_like(scores, dtype=float)
            pos_mask = logits >= 0
            neg_mask = ~pos_mask

            # y * log(1 + e^-logit) + (1 - y) * (logit + log(1 + e^-logit))
            loss[pos_mask] = labels[pos_mask] * np.log1p(np.exp(-logits[pos_mask])) + (
                1.0 - labels[pos_mask]
            ) * (logits[pos_mask] + np.log1p(np.exp(-logits[pos_mask])))

            # y * (-logit + log(1 + e^logit)) + (1 - y) * log(1 + e^logit)
            loss[neg_mask] = labels[neg_mask] * (
                -logits[neg_mask] + np.log1p(np.exp(logits[neg_mask]))
            ) + (1.0 - labels[neg_mask]) * np.log1p(np.exp(logits[neg_mask]))

            return float(np.mean(loss))

        # Initialize parameters: negative correlation expected between score and class logit
        initial_params = np.array([-1.0, float(np.mean(scores))])
        res = minimize(neg_log_likelihood, initial_params, method="Nelder-Mead")

        if res.success:
            self.A, self.B = res.x
            self.is_fitted = True
            print(f"✓ Platt calibration fitted! A: {self.A:.4f}, B: {self.B:.4f}")
        else:
            # Fallback estimation based on class separation statistics
            self.A = -1.0 / (np.std(scores) + 1e-8)
            self.B = -self.A * np.mean(scores)
            self.is_fitted = True

    def calibrate(self, scores: Any) -> Any:
        """Applies fitted logistic calibration to scores (Tensor or Array) returning values in [0, 1]."""
        if not self.is_fitted:
            # Simple soft sigmoid fallback if calibration is not executed yet
            if isinstance(scores, torch.Tensor):
                return torch.sigmoid(scores - 1.0)
            else:
                return 1.0 / (1.0 + np.exp(-(scores - 1.0)))

        if isinstance(scores, torch.Tensor):
            return 1.0 / (1.0 + torch.exp(self.A * scores + self.B))
        else:
            return 1.0 / (1.0 + np.exp(self.A * scores + self.B))


class AnomalyEnsemble(BaseAnomalyModel):
    """
    Weighted Late Fusion Ensemble for Anomaly Detection.

    Combines anomaly scores and anomaly maps from multiple models (e.g. PatchCore and EfficientAD).
    The weights are learned dynamically by minimizing validation Binary Cross Entropy loss
    computed over Platt-calibrated probability spaces.
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
        self.calibrators = {name: PlattCalibrator() for name in models.keys()}

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Runs forward inference across all models in the ensemble and fuses results.

        Args:
            x: Input torch.Tensor.

        Returns:
            A dictionary containing:
                - 'score': The final fused calibrated anomaly probability (float tensor).
                - 'anomaly_map': The fused probability-calibrated anomaly map.
                - 'model_scores': Dictionary of individual model scores.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name, model in self.models.items():
            results[name] = model(x)

        # Extract individual scores
        scores: Dict[str, torch.Tensor] = {
            name: (
                res.get("pred_score", res.get("score"))
                if isinstance(res, dict)
                else res
            )
            for name, res in results.items()
        }

        # Weighted fusion of calibrated scores
        fused_score = torch.zeros_like(list(scores.values())[0])
        for name, score in scores.items():
            cal_score = self.calibrators[name].calibrate(score)
            fused_score += self._learned_weights[name] * cal_score

        # Weighted fusion of calibrated anomaly maps
        fused_map = torch.zeros_like(list(results.values())[0]["anomaly_map"])
        for name, res in results.items():
            cal_map = self.calibrators[name].calibrate(res["anomaly_map"])
            fused_map += self._learned_weights[name] * cal_map

        return {"score": fused_score, "anomaly_map": fused_map, "model_scores": scores}

    def optimize_weights(
        self,
        val_scores_m1: np.ndarray,
        val_scores_m2: np.ndarray,
        val_labels: np.ndarray,
    ) -> None:
        """
        Optimizes score fusion weights by minimizing Binary Cross Entropy loss on a validation set.
        Calibrates model scores via Platt scaling to ensure statistical validity of probability inputs.

        Uses scipy.optimize.minimize to find weights w1, w2 that minimize BCE.

        Args:
            val_scores_m1: Array of validation anomaly scores from model 1.
            val_scores_m2: Array of validation anomaly scores from model 2.
            val_labels: Binary ground-truth labels (0 for normal, 1 for anomaly).
        """
        model_names = list(self.models.keys())
        if len(model_names) >= 2:
            self.calibrators[model_names[0]].fit(val_scores_m1, val_labels)
            self.calibrators[model_names[1]].fit(val_scores_m2, val_labels)

            cal_val_m1 = self.calibrators[model_names[0]].calibrate(val_scores_m1)
            cal_val_m2 = self.calibrators[model_names[1]].calibrate(val_scores_m2)
        else:
            cal_val_m1 = val_scores_m1
            cal_val_m2 = val_scores_m2

        def bce_loss(weights: np.ndarray) -> float:
            w1, w2 = weights
            fused_prob = w1 * cal_val_m1 + w2 * cal_val_m2

            # Clip probabilities to avoid log(0)
            probs = np.clip(fused_prob, 1e-7, 1.0 - 1e-7)
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
