"""
GDXray Radiographic Weld Scan Loader for AutoWeld-Vision.

Loads radiographic X-ray inspection frames from the GDXray database (Welds category).
Parses bounding boxes from XML annotation files to build pixel-level defect masks.
"""

import os
import glob
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, List, Dict, Any


class GDXrayWeldDataset(Dataset):
    """
    GDXray Radiographic Weld Scan Dataset Loader.
    Loads radiographic X-ray visual inspection frames from the GDXray Welds category.
    Parses bounding boxes or masks from matching XML annotation files.
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
                f"GDXray Weld Dataset root directory '{root}' not found.\n"
                f"Please download the GDXray dataset Welds series from "
                f"http://dmery.ing.puc.cl/index.php/material/gdxray/ and extract it."
            )

        self._load_samples()

    def _load_samples(self) -> None:
        # GDXray Welds series directories are W0001, W0002, etc.
        weld_series_dirs = glob.glob(os.path.join(self.root, "W[0-9]*"))

        for series_dir in sorted(weld_series_dirs):
            series_name = os.path.basename(series_dir)

            # Find all PNG/JPG images in the series
            img_paths = glob.glob(os.path.join(series_dir, "*.png")) + glob.glob(
                os.path.join(series_dir, "*.jpg")
            )
            for img_path in sorted(img_paths):
                filename = os.path.splitext(os.path.basename(img_path))[0]

                # Check for matching XML bounding box annotations
                xml_path = os.path.join(series_dir, f"{filename}.xml")
                has_defect = False
                boxes = []

                if os.path.exists(xml_path):
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        for obj in root.findall("object"):
                            name_el = obj.find("name")
                            if name_el is not None and name_el.text is not None:
                                name = name_el.text
                                if (
                                    "defect" in name.lower()
                                    or "anomaly" in name.lower()
                                ):
                                    has_defect = True
                                    bbox = obj.find("bndbox")
                                    if bbox is not None:
                                        xmin_el = bbox.find("xmin")
                                        ymin_el = bbox.find("ymin")
                                        xmax_el = bbox.find("xmax")
                                        ymax_el = bbox.find("ymax")
                                        if (
                                            xmin_el is not None
                                            and xmin_el.text is not None
                                            and ymin_el is not None
                                            and ymin_el.text is not None
                                            and xmax_el is not None
                                            and xmax_el.text is not None
                                            and ymax_el is not None
                                            and ymax_el.text is not None
                                        ):
                                            xmin = int(xmin_el.text)
                                            ymin = int(ymin_el.text)
                                            xmax = int(xmax_el.text)
                                            ymax = int(ymax_el.text)
                                            boxes.append([xmin, ymin, xmax, ymax])
                    except Exception:
                        pass

                # Simple determinism for train/test split: W0001-W0008 are train, rest are test
                try:
                    series_num = int(series_name[1:])
                except ValueError:
                    series_num = 1

                is_train = series_num <= 8

                if (self.split == "train" and is_train) or (
                    self.split == "test" and not is_train
                ):
                    self.samples.append(
                        {
                            "image_path": img_path,
                            "label": 1 if has_defect else 0,
                            "xml_path": xml_path if os.path.exists(xml_path) else None,
                            "boxes": boxes,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        w, h = image.size

        # Create binary ground truth mask from bounding boxes
        mask = torch.zeros((1, h, w))
        for box in sample["boxes"]:
            xmin, ymin, xmax, ymax = box
            # Clip bounds
            xmin = max(0, min(xmin, w - 1))
            xmax = max(0, min(xmax, w - 1))
            ymin = max(0, min(ymin, h - 1))
            ymax = max(0, min(ymax, h - 1))
            mask[0, ymin:ymax, xmin:xmax] = 1.0

        if self.transform:
            image = self.transform(image)
            # Custom resizing for masks
            import torchvision.transforms.functional as F_t

            # Convert to PIL Image for standard resize then back to Tensor
            mask_pil = Image.fromarray((mask[0].numpy() * 255).astype("uint8"))
            mask_resized = F_t.resize(
                mask_pil,
                (image.shape[1], image.shape[2]),
                interpolation=Image.Resampling.NEAREST,
            )
            mask = F_t.to_tensor(mask_resized)

        return {
            "image": image,
            "label": sample["label"],
            "path": sample["image_path"],
            "mask": mask,
            "meta": {
                "dataset": "GDXray",
                "split": self.split,
                "boxes": sample["boxes"],
            },
        }
