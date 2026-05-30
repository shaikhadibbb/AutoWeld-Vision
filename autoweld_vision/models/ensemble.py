"""
Late Fusion Ensemble and Defect Gating for Industrial Anomaly Detection.

This module implements AnomalyEnsemble which fuses the anomaly scores and anomaly maps
from PatchCore and EfficientAD. The score fusion weights are learned programmatically
by minimizing binary cross-entropy validation loss on a held-out validation split.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from scipy.optimize import minimize
from .base import BaseAnomalyModel


class AnomalyEnsemble(BaseAnomalyModel):
    """
    Weighted Late Fusion Ensemble for Anomaly Detection.

    Combines anomaly scores and anomaly maps from multiple models (e.g. PatchCore and EfficientAD).
    The weights are learned by minimizing validation binary cross-entropy loss.
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
            name: res["score"] for name, res in results.items()
        }

        # Weighted fusion of scores
        # We handle arbitrary keys but specifically patchcore and efficientad if present
        fused_score = torch.zeros_like(list(scores.values())[0])
        for name, score in scores.items():
            fused_score += self._learned_weights[name] * score

        # Weighted fusion of anomaly maps
        fused_map = torch.zeros_like(list(results.values())[0]["anomaly_map"])
        for name, res in results.items():
            fused_map += self._learned_weights[name] * res["anomaly_map"]

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

        # Sigmoid activation helper to bound scores to [0, 1] for BCE loss
        def sigmoid(v: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-v))

        def bce_loss(weights: np.ndarray) -> float:
            w1, w2 = weights
            fused = w1 * val_scores_m1 + w2 * val_scores_m2
            probs = sigmoid(fused)
            # Clip probabilities to avoid log(0)
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
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
            model_names = list(self.models.keys())
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

    Routes input images to specialist anomaly detection models based on coarse detection.
    """

    def __init__(self, specialists: Dict[str, nn.Module]) -> None:
        """
        Initializes the DefectRouter.

        Args:
            specialists: A dictionary mapping defect categories to their specialized models.
        """
        self.specialists = specialists
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128, len(specialists)),
        )

    def route(self, image: torch.Tensor, coarse_map: torch.Tensor) -> str:
        """
        Routes the image to the appropriate specialist model.

        Args:
            image: The input image tensor.
            coarse_map: The initial coarse anomaly map tensor.

        Returns:
            The name of the chosen specialist defect model.
        """
        with torch.no_grad():
            logits = self.classifier(coarse_map)
            defect_idx = int(torch.argmax(logits, dim=1).item())
            defect_type = list(self.specialists.keys())[defect_idx]
        return defect_type
