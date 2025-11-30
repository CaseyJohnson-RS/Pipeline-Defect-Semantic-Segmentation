import os
import cv2
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Быстрый датасет для бинарной сегментации.
    """
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        img_size: Tuple[int, int] = (700, 500),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        # cv2.resize expects dsize=(width, height)
        self.img_size = img_size[::-1]
        self.normalize = bool(normalize)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

        self.images = sorted(
            [os.path.join(images_dir, f) for f in os.listdir(images_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        )
        self.masks = sorted(
            [os.path.join(masks_dir, f) for f in os.listdir(masks_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        )

        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Кол-во изображений ({len(self.images)}) ≠ масок ({len(self.masks)})"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        # --------- изображение ---------
        img = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)  # BGR HWC
        if img is None:
            raise RuntimeError(f"Не удалось прочитать изображение {self.images[idx]}")
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img[:, :, ::-1])           # RGB + контикуально
        img = torch.from_numpy(img).float().permute(2, 0, 1).div_(255.0)  # CHW, 0-1
        if self.normalize:
            # normalize in-place (will broadcast)
            img = (img - self.mean) / self.std
        
        # --------- маска ---------
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Не удалось прочитать маску {self.masks[idx]}")
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # нормализуем 0-255 → 0-1, если нужно
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # 1HW

        return img, mask