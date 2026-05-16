import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import numpy as np
from .base import BaseAnomalyModel

class AnomalyEnsemble(BaseAnomalyModel):
    """
    Priority 1.1: Late Fusion Ensemble.
    Combines predictions from multiple anomaly detectors (PatchCore, EfficientAD, Dinomaly).
    Provides uncertainty quantification and ensemble-boosted AUROC.
    """
    def __init__(self, models: Dict[str, nn.Module], fusion_type: str = 'weighted'):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.fusion_type = fusion_type
        
        # Meta-learner for 'learned' fusion
        if fusion_type == 'learned':
            self.meta_mlp = nn.Sequential(
                nn.Linear(len(models), 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            
        # Default weights (can be updated via logistic regression on validation set)
        self.weights = {name: 1.0/len(models) for name in models.keys()}

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        results = {}
        for name, model in self.models.items():
            results[name] = model(x)
            
        scores = torch.stack([res['score'] for res in results.values()], dim=1) # [B, N, 1]
        scores = scores.squeeze(-1) # [B, N]
        
        if self.fusion_type == 'weighted':
            # Compute weighted average
            w_tensor = torch.tensor([self.weights[n] for n in self.models.keys()], device=x.device)
            final_score = (scores * w_tensor).sum(dim=1, keepdim=True)
            
        elif self.fusion_type == 'learned':
            final_score = self.meta_mlp(scores)
            
        elif self.fusion_type == 'uncertainty':
            # Lower variance across models = higher confidence (weight)
            # This is a heuristic: models that agree are weighted higher
            variances = torch.var(scores, dim=1, keepdim=True)
            weights = 1.0 / (variances + 1e-6)
            weights = weights / weights.sum()
            final_score = (scores * weights).sum(dim=1, keepdim=True)
        
        # Uncertainty quantification: Variance across ensemble predictions
        ensemble_variance = torch.var(scores, dim=1, keepdim=True)
        
        # Fuse anomaly maps (simple average for now)
        fused_map = torch.stack([res['anomaly_map'] for res in results.values()], dim=1).mean(dim=1)
        
        return {
            'score': final_score,
            'anomaly_map': fused_map,
            'uncertainty': ensemble_variance,
            'model_contributions': {n: res['score'] for n, res in results.items()}
        }

class DefectRouter:
    """
    Priority 1.2: Defect-Type-Specific Routing.
    Routes images to specialist models based on coarse detection characteristics.
    """
    def __init__(self, specialists: Dict[str, nn.Module]):
        self.specialists = specialists
        # Small CNN to classify defect type from anomaly map features
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128, len(specialists))
        )

    def route(self, image: torch.Tensor, coarse_map: torch.Tensor) -> str:
        # Predict defect type from the coarse anomaly map
        with torch.no_grad():
            logits = self.classifier(coarse_map)
            defect_idx = torch.argmax(logits, dim=1).item()
            defect_type = list(self.specialists.keys())[defect_idx]
        return defect_type
