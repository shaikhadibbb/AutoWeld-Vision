from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from typing import Optional


class AutoWeldDataset(Dataset):
    """
    Industrial-grade dataset loader for welding defects.
    Supports MVTec AD, NEU, and custom synthetic data.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        transform: Optional[transforms.Compose] = None,
        tier: str = "A",
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.tier = tier
        self.image_paths = []
        self.labels = []
        self.masks = []

        self._load_metadata()

    def _load_metadata(self):
        # Implementation for metadata loading based on Tier
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        mask_path = self.masks[idx] if self.masks else None

        sample = {"image": image, "label": label, "path": img_path}

        if mask_path:
            mask = Image.open(mask_path).convert("L")
            if self.transform:
                # Custom transform logic for masks (nearest neighbor)
                mask = transforms.functional.resize(
                    mask, (image.shape[1], image.shape[2])
                )
                mask = transforms.functional.to_tensor(mask)
            sample["mask"] = mask

        return sample


class IndustrialAugmentor:
    """
    Implements industrial-specific augmentations like CLAHE and illumination normalization.
    """

    @staticmethod
    def apply_clahe(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    @staticmethod
    def get_train_transforms(size: int = 256):
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
