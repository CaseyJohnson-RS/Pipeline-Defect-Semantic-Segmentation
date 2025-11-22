import cv2
import random
import numpy as np
import albumentations as A
from tqdm import tqdm
import torch
from pathlib import Path
from src import set_seed
from typing import Optional

def find_mask_by_image_name(mask_dir: Path, img_file: str) -> Optional[Path]:
    """
    Finds mask file by the first 6 characters of image name.
    Checks for .png masks.
    
    Args:
        mask_dir: directory with masks
        img_file: image filename (e.g., "123456_aug.png")
    
    Returns:
        Path to mask or None if not found
    """
    if not mask_dir.exists():
        return None
        
    base_name = Path(img_file).stem[:6]
    candidates = list(mask_dir.glob(f"{base_name}*.png"))
    
    if not candidates:
        return None

    # Return mask with shortest name (without extra suffixes)
    candidates.sort(key=lambda p: len(p.name))
    return candidates[0]

def augment_and_save(
    image_path: str, 
    mask_path: str, 
    out_image_dir: Path, 
    out_mask_dir: Path, 
    filename: str,
    transform: A.Compose,
    n_aug: int,
    seed: int
):
    """
    Performs augmentation of image and mask and saves results.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error reading: {image_path} / {mask_path}")
        return

    if image.shape[:2] != mask.shape[:2]:
        print(f"Size mismatch: {image.shape[:2]} vs {mask.shape[:2]}")
        return

    base_name = Path(filename).stem

    for i in range(n_aug):
        # Use own seed for each augmentation
        local_seed = seed + i
        random.seed(local_seed)
        np.random.seed(local_seed)

        augmented = transform(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        cv2.imwrite(str(out_image_dir / f"{base_name}_aug{i}.png"), aug_image)
        cv2.imwrite(str(out_mask_dir / f"{base_name}_aug{i}.png"), aug_mask)

def augmentation_segmentation_ds(dataset_name: str, n_aug: int = 3, seed: int = 42):
    """
    Performs augmentation of images and masks for a specified dataset.
    
    Args:
        dataset_name (str): dataset name (e.g., "PipeSegmentation")
        n_aug (int): number of augmentations per image
        seed (int): random seed for reproducibility
    """
    
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    SPLITS = ["train", "val"]
    base_dir = Path(f"datasets/{dataset_name}")
    base_output_dir = Path(f"datasets/{dataset_name}_augmented")

    # Check if input directory exists
    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    image_dirs = {split: base_dir / "images" / split for split in SPLITS}
    mask_dirs = {split: base_dir / "masks" / split for split in SPLITS}
    output_image_dirs = {split: base_output_dir / "images" / split for split in SPLITS}
    output_mask_dirs = {split: base_output_dir / "masks" / split for split in SPLITS}

    # Create output directories
    for split in SPLITS:
        output_image_dirs[split].mkdir(parents=True, exist_ok=True)
        output_mask_dirs[split].mkdir(parents=True, exist_ok=True)

    transform = A.Compose([
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.01,), p=1.0),
            A.ISONoise(p=1.0)
        ], p=0.5),
        A.Rotate(limit=10, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8)
    ], additional_targets={"mask": "mask"})

    for split in SPLITS:
        image_files = list(image_dirs[split].glob("*.png"))
        
        if not image_files:
            print(f"Warning: no .png images found in {image_dirs[split]}")
            continue
            
        for img_file in tqdm(image_files, desc=f"Augmenting {split}", unit="img"):
            mask_path = find_mask_by_image_name(mask_dirs[split], img_file.name)
            
            if mask_path:
                augment_and_save(
                    str(img_file),
                    str(mask_path),
                    output_image_dirs[split],
                    output_mask_dirs[split],
                    img_file.name,
                    transform,
                    n_aug,
                    seed
                )
            else:
                base_name = img_file.stem[:6]
                print(f"Mask not found for image '{img_file.name}' (search by base: '{base_name}')")

    print(f"Augmentation completed. Results saved in: {base_output_dir}")