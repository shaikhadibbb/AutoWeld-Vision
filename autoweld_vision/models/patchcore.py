from .base import BaseAnomalyModel
from .registry import ModelRegistry
import torch
from anomalib.models.image.patchcore import Patchcore

@ModelRegistry.register("patchcore")
class PatchCoreModel(BaseAnomalyModel):
    def __init__(self, backbone='wide_resnet50_2', layers=['layer2', 'layer3']):
        super().__init__()
        # Use the real Anomalib Patchcore
        self.model = Patchcore(backbone=backbone)
        print(f"Initialized real PatchCore with {backbone}")

    def forward(self, x):
        # Anomalib models return a dictionary with 'anomaly_map' and 'pred_score'
        # Note: They usually expect a Lightning engine, but we can call torch_model
        # For simplicity in this demo, we'll use the torch_model directly if available
        # Or just mock the output for the server if it's not trained.
        
        # In a real demo, we'd load weights here.
        return self.model(x)
