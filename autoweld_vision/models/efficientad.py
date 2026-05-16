from .base import BaseAnomalyModel
from .registry import ModelRegistry
import torch
from anomalib.models.image.efficient_ad import EfficientAd

@ModelRegistry.register("efficientad")
class EfficientADModel(BaseAnomalyModel):
    def __init__(self):
        super().__init__()
        self.model = EfficientAd()
        print("Initialized real EfficientAD")

    def forward(self, x):
        return self.model(x)
