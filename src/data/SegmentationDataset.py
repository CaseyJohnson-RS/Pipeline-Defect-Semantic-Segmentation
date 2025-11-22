from torch.utils.data import Dataset
from glob import glob
from torchvision import transforms
from PIL import Image
import os
import cv2
import torch
from typing import Tuple


class SegmentationDataset(Dataset):
    """Custom dataset for binary segmentation with soft mask support."""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        img_size: Tuple[int, int] = (700, 500),
    ):
        # --- basic directory checks ---
        for path, name in ((images_dir, "images"), (masks_dir, "masks")):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"{name} directory does not exist: {path}")
            if not os.listdir(path):
                raise ValueError(f"{name} directory is empty: {path}")

        self.images = sorted(glob(os.path.join(images_dir, "*")))
        self.masks = sorted(glob(os.path.join(masks_dir, "*")))

        if not self.images:
            raise ValueError("No image files found in " + images_dir)
        if not self.masks:
            raise ValueError("No mask files found in " + masks_dir)
        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Amount of images ({len(self.images)}) != masks ({len(self.masks)})"
            )

        self.img_size = img_size

        # Transforms for images (PIL)
        self.transform_img = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Load image via PIL, soft mask via OpenCV (preserves values)."""
        # Load image
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform_img(img)
        
        # Load soft mask via OpenCV (doesn't change values)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {self.masks[idx]}")
        
        # Resize with NEAREST (preserves original values 0-255)
        mask = cv2.resize(mask, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        # Normalize [0, 255] -> [0, 1]
        mask = torch.from_numpy(mask).float() / 255.0
        
        # Channel: [H, W] -> [1, H, W]
        mask = mask.unsqueeze(0)
        
        return img, mask