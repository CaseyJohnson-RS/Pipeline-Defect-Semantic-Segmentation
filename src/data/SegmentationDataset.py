from torch.utils.data import Dataset
from glob import glob
from torchvision import transforms
from PIL import Image
import os
import cv2
import torch
from typing import Tuple


class SegmentationDataset(Dataset):
    """Correct dataset for soft masks without value distortion."""

    def __init__(
        self, 
        images_dir: str, 
        masks_dir: str, 
        img_size: Tuple[int, int] = (700, 500),
    ):
        self.images = sorted(glob(os.path.join(images_dir, "*")))
        self.masks = sorted(glob(os.path.join(masks_dir, "*")))
        self.img_size = img_size

        # Transform для изображений (PIL)
        self.transform_img = transforms.Compose([
            transforms.Resize(self.img_size), 
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Load image via PIL, soft mask via OpenCV (preserves values)."""
        # Загрузка изображения
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform_img(img)
        
        # Загрузка soft маски через OpenCV (не меняет значения)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {self.masks[idx]}")
        
        # Resize с NEAREST (сохраняет исходные значения 0-255)
        mask = cv2.resize(mask, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        # Нормализация [0, 255] -> [0, 1]
        mask = torch.from_numpy(mask).float() / 255.0
        
        # Канал: [H, W] -> [1, H, W]
        mask = mask.unsqueeze(0)
        
        return img, mask