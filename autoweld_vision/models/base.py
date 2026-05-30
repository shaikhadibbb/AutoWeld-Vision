import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAnomalyModel(nn.Module, ABC):
    """Base class for anomaly detection models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def predict(self, x):
        """Standard prediction wrapper."""
        return self.forward(x)
