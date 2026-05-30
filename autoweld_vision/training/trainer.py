import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from autoweld_vision.models.registry import ModelRegistry
from autoweld_vision.data.dataset import AutoWeldDataset, IndustrialAugmentor
from typing import Dict, Any


class AutoWeldTrainer(pl.LightningModule):
    def __init__(self, model_name: str, model_config: Dict[str, Any], lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = ModelRegistry.get_model(model_name, model_config)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        # Most anomaly detection models (unsupervised) use normal images
        # We might compute reconstruction loss or feature distance
        outputs = self.model(images)
        loss = outputs.mean()  # Placeholder
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]
        scores = self.model(images)
        # Log validation metrics like AUROC
        self.log("val_score", scores.mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_pipeline(config: Dict[str, Any]):
    """
    Orchestrates the training process using Hydra configs.
    """
    dataset = AutoWeldDataset(
        root_dir=config["data"]["root"],
        mode="train",
        transform=IndustrialAugmentor.get_train_transforms(),
        tier=config["data"]["tier"],
    )
    train_loader = DataLoader(
        dataset, batch_size=config["train"]["batch_size"], shuffle=True
    )

    trainer_pl = pl.Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="auto",
        devices=1,
        precision=16 if config["train"]["mixed_precision"] else 32,
    )

    model = AutoWeldTrainer(
        model_name=config["model"]["name"], model_config=config["model"]["params"]
    )

    trainer_pl.fit(model, train_loader)
