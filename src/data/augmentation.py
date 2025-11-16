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
    Находит файл маски по первым 6 символам имени изображения.
    Проверяет .png маски.
    
    Args:
        mask_dir: директория с масками
        img_file: имя файла изображения (например, "123456_aug.png")
    
    Returns:
        Path к маске или None, если не найдено
    """
    if not mask_dir.exists():
        return None
        
    base_name = Path(img_file).stem[:6]
    candidates = list(mask_dir.glob(f"{base_name}*.png"))
    
    if not candidates:
        return None

    # Возвращаем маску с самым коротким именем (без дополнительных суффиксов)
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
    Выполняет аугментацию изображения и маски и сохраняет результаты.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Ошибка чтения: {image_path} / {mask_path}")
        return

    if image.shape[:2] != mask.shape[:2]:
        print(f"Несовпадение размеров: {image.shape[:2]} vs {mask.shape[:2]}")
        return

    base_name = Path(filename).stem

    for i in range(n_aug):
        # Используем собственный seed для каждого augment'а
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
    Выполняет аугментацию изображений и масок для указанного датасета.
    
    Args:
        dataset_name (str): название датасета (например, "PipeSegmentation")
        n_aug (int): количество аугментаций на одно изображение
        seed (int): случайное зерно для воспроизводимости
    """
    
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    SPLITS = ["train", "val"]
    base_dir = Path(f"datasets/{dataset_name}")
    base_output_dir = Path(f"datasets/{dataset_name}_augmented")

    # Проверка существования входной директории
    if not base_dir.exists():
        raise FileNotFoundError(f"Директория датасета не найдена: {base_dir}")

    image_dirs = {split: base_dir / "images" / split for split in SPLITS}
    mask_dirs = {split: base_dir / "masks" / split for split in SPLITS}
    output_image_dirs = {split: base_output_dir / "images" / split for split in SPLITS}
    output_mask_dirs = {split: base_output_dir / "masks" / split for split in SPLITS}

    # Создаём выходные директории
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
            print(f"Предупреждение: в {image_dirs[split]} не найдено .png изображений")
            continue
            
        for img_file in tqdm(image_files, desc=f"Аугментация {split}", unit="img"):
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
                print(f"Маска не найдена для изображения '{img_file.name}' (поиск по базе: '{base_name}')")

    print(f"Аугментация завершена. Результаты сохранены в: {base_output_dir}")