"""
Unit tests for the AnomalyEnsemble score fusion module.
"""

import pytest
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
        return {
            "score": self.score.repeat(batch_size, 1),
            "anomaly_map": anomaly_map
        }

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
