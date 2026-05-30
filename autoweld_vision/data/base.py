from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseIndustrialDataset(Dataset, ABC):
    """Base class for industrial anomaly detection datasets."""

    def __init__(self, root, category, split="train", transform=None):
        self.root = root
        self.category = category
        self.split = split
        self.transform = transform
        self.samples = []

    def __len__(self):
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx):
        pass
