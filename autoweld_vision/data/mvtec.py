import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, List, Dict, Any
from torchvision import transforms as T

class BaseAnomalyDataset(Dataset):
    """Unified interface for all anomaly detection datasets."""
    def __init__(self, root: str, category: str, split: str = 'train', transform: Optional[Any] = None):
        self.root = root
        self.category = category
        self.split = split
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        target = {
            'image': image,
            'label': sample['label'],
            'path': sample['image_path'],
            'meta': {'category': self.category, 'split': self.split}
        }
        
        if sample['mask_path']:
            mask = Image.open(sample['mask_path']).convert('L')
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, T.Resize):
                        mask = mask.resize(t.size, Image.NEAREST)
            target['mask'] = T.ToTensor()(mask)
        else:
            target['mask'] = torch.zeros((1, 256, 256))
            
        return target

class MVTecADLoader(BaseAnomalyDataset):
    """Standard MVTec AD Loader."""
    def __init__(self, root: str, category: str, split: str = 'train', **kwargs):
        super().__init__(root, category, split, **kwargs)
        self._load_samples()

    def _load_samples(self):
        split_dir = os.path.join(self.root, self.category, self.split)
        
        if self.split == 'train':
            image_dir = os.path.join(split_dir, 'good')
            for img_path in sorted(glob.glob(os.path.join(image_dir, "*.png"))):
                self.samples.append({'image_path': img_path, 'label': 0, 'mask_path': None})
        else:
            defect_types = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            for dt in defect_types:
                label = 0 if dt == 'good' else 1
                dt_dir = os.path.join(split_dir, dt)
                # Handle lighting in MVTec AD 2 if needed
                for img_path in sorted(glob.glob(os.path.join(dt_dir, "**/*.png"), recursive=True)):
                    mask_path = None
                    if label == 1:
                        # Guess mask path (root/category/ground_truth/defect_type/filename_mask.png)
                        rel_path = os.path.relpath(img_path, dt_dir)
                        mask_name = os.path.splitext(os.path.basename(img_path))[0] + "_mask.png"
                        mask_path = os.path.join(self.root, self.category, 'ground_truth', dt, os.path.dirname(rel_path), mask_name)
                        if not os.path.exists(mask_path):
                            mask_path = None
                    self.samples.append({'image_path': img_path, 'label': label, 'mask_path': mask_path})
