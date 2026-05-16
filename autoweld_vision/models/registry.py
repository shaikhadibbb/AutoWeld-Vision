import torch
import torch.nn as nn
from typing import Dict, Any, Type

class ModelRegistry:
    """
    Central registry for all AutoWeld-Vision anomaly detection models.
    Supports dynamic instantiation via config files.
    """
    _models = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_class: Type):
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, name: str, config: Dict[str, Any] = None) -> nn.Module:
        config = config or {}
        if name not in cls._models:
            # Attempt to import all models to trigger registration
            from . import patchcore, efficientad, dinomaly, ensemble
            
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry. Available: {list(cls._models.keys())}")
        
        return cls._models[name](**config)
