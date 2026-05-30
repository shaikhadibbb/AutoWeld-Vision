"""
Unit tests for the AnomalyEnsemble score fusion module.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from autoweld_vision.models.ensemble import AnomalyEnsemble


class MockAnomalyModel(nn.Module):
    """Mock model returning controllable predefined scores and maps."""

    def __init__(self, score: float, val: float = 0.0) -> None:
        super().__init__()
        self.score = torch.tensor([[score]])
        self.val = val

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        batch_size = x.shape[0]
        # Return a uniform mock map
        anomaly_map = torch.ones(batch_size, 1, 256, 256) * self.val
        return {"score": self.score.repeat(batch_size, 1), "anomaly_map": anomaly_map}


def test_ensemble_bounds_and_monotonicity() -> None:
    """Verifies score boundedness in [0, 1] and monotonicity with respect to model inputs."""
    m1 = MockAnomalyModel(0.2, val=0.1)
    m2 = MockAnomalyModel(0.8, val=0.7)

    ensemble = AnomalyEnsemble({"m1": m1, "m2": m2})
    ensemble._learned_weights = {"m1": 0.4, "m2": 0.6}

    x = torch.zeros(1, 3, 256, 256)
    out = ensemble(x)

    # 1. Assert score outputs
    fused_score = out["score"].item()
    assert 0.0 <= fused_score <= 1.0
    # Expected: 0.4 * 0.2 + 0.6 * 0.8 = 0.08 + 0.48 = 0.56
    assert abs(fused_score - 0.56) < 1e-5

    # 2. Test monotonicity
    # If we increase m1's score, fused score must increase
    m1_higher = MockAnomalyModel(0.4, val=0.1)
    ensemble_higher = AnomalyEnsemble({"m1": m1_higher, "m2": m2})
    ensemble_higher._learned_weights = {"m1": 0.4, "m2": 0.6}
    out_higher = ensemble_higher(x)
    assert out_higher["score"].item() > fused_score


def test_ensemble_optimize_weights() -> None:
    """Verifies that optimize_weights correctly solves SLSQP and learns valid fusion weights."""
    m1 = MockAnomalyModel(0.2, val=0.1)
    m2 = MockAnomalyModel(0.8, val=0.7)
    ensemble = AnomalyEnsemble({"m1": m1, "m2": m2})

    # Generate some validation-like scores
    np.random.seed(42)
    val_scores_m1 = np.random.uniform(0.1, 0.4, 20)
    val_scores_m2 = np.random.uniform(0.6, 0.9, 20)

    # Let normal labels correspond to low scores and anomalies to high
    val_labels = np.array([0] * 10 + [1] * 10)

    # Optimize weights
    ensemble.optimize_weights(val_scores_m1, val_scores_m2, val_labels)

    # Assert weights sum to 1.0 and are within bounds
    assert abs(ensemble.w1 + ensemble.w2 - 1.0) < 1e-4
    assert 0.0 <= ensemble.w1 <= 1.0
    assert 0.0 <= ensemble.w2 <= 1.0
    assert "m1" in ensemble._learned_weights
    assert "m2" in ensemble._learned_weights
    assert abs(ensemble._learned_weights["m1"] - ensemble.w1) < 1e-6
    assert abs(ensemble._learned_weights["m2"] - ensemble.w2) < 1e-6


def test_defect_router() -> None:
    """Verifies that DefectRouter routes image tensors correctly based on classifier logits."""
    from autoweld_vision.models.ensemble import DefectRouter

    s1 = MockAnomalyModel(0.1)
    s2 = MockAnomalyModel(0.9)
    specialists = {"sp1": s1, "sp2": s2}

    router = DefectRouter(specialists)

    # Create mock inputs
    image = torch.zeros(1, 3, 256, 256)
    coarse_map = torch.zeros(1, 1, 256, 256)

    route_choice = router.route(image, coarse_map)
    assert route_choice in specialists.keys()
