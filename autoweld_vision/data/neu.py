"""
NEU Surface Defect PyTorch Loader for AutoWeld-Vision.

Loads the Northeastern University (NEU) steel surface defect database.
Categorizes rolled-in scale, inclusions, patches, pitted surfaces, scratches, and cracks.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, List, Dict, Any


class NEUSurfaceDefectDataset(Dataset):
    """
    NEU Steel Surface Defect Database Loader.
    Supports six typical defect categories:
    - Rolled-in Scale (rolled-in_scale)
    - Inclusion (inclusion)
    - Patches (patches)
    - Pitted Surface (pitted_surface)
    - Scratches (scratches)
    - Cracks (cracks)
    """

    def __init__(
        self, root: str, split: str = "train", transform: Optional[Any] = None
    ) -> None:
        self.root = root
        self.split = split
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []

        if not os.path.exists(root):
            raise FileNotFoundError(
                f"NEU Surface Defect root directory '{root}' not found.\n"
                f"Please download the NEU dataset from http://faculty.neu.edu.cn/ye/NEU-CLS.html "
                f"and extract it to the designated root path."
            )

        self._load_samples()

    def _load_samples(self) -> None:
        defect_classes = [
            "rolled-in_scale",
            "inclusion",
            "patches",
            "pitted_surface",
            "scratches",
            "cracks",
        ]

        # Scan recursively for images following the class_name_number.jpg naming format
        for class_name in defect_classes:
            pattern = os.path.join(self.root, "**", f"{class_name}_*.jpg")
            for img_path in sorted(glob.glob(pattern, recursive=True)):
                filename = os.path.basename(img_path)
                try:
                    num_part = int(filename.split("_")[-1].split(".")[0])
                except ValueError:
                    num_part = 0

                # 300 images per class. Assign first 240 (80%) to train, remaining to test
                is_train = num_part <= 240

                if (self.split == "train" and is_train) or (
                    self.split == "test" and not is_train
                ):
                    self.samples.append(
                        {
                            "image_path": img_path,
                            "label": 1,  # 1 denotes defect presence
                            "defect_type": class_name,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "path": sample["image_path"],
            "defect_type": sample["defect_type"],
            "mask": torch.zeros((1, 256, 256)),  # Class level classification
            "meta": {"dataset": "NEU", "split": self.split},
        }
