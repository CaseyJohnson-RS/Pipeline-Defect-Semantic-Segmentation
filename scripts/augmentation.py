import os
import cv2
import random
import numpy as np
import albumentations as A
from tqdm import tqdm
import torch
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

def run_augmentation(dataset_name: str, n_aug: int = 3, seed: int = 42):
    """
    Выполняет аугментацию изображений и масок для указанного датасета.
    
    Args:
        dataset_name (str): название датасета (например, "PipeSegmentation")
        n_aug (int): количество аугментаций на одно изображение
        seed (int): случайное зерно для воспроизводимости
    """
    
    # ==========================================================
    # Фиксируем все random seed'ы
    # ==========================================================
    SEED = seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Для повторяемости при работе DataLoader/torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ==========================================================
    # Настройки путей
    # ==========================================================
    SPLITS = ["train", "val"]

    BASE_DIR = f"datasets/{dataset_name}"
    BASE_OUTPUT_DIR = f"datasets/{dataset_name}_augmented"

    IMAGE_DIRS = {split: os.path.join(BASE_DIR, "images", split) for split in SPLITS}
    MASK_DIRS = {split: os.path.join(BASE_DIR, "masks", split) for split in SPLITS}
    OUTPUT_IMAGE_DIRS = {split: os.path.join(BASE_OUTPUT_DIR, "images", split) for split in SPLITS}
    OUTPUT_MASK_DIRS = {split: os.path.join(BASE_OUTPUT_DIR, "masks", split) for split in SPLITS}


    # Создаём выходные директории
    for split in SPLITS:
        os.makedirs(OUTPUT_IMAGE_DIRS[split], exist_ok=True)
        os.makedirs(OUTPUT_MASK_DIRS[split], exist_ok=True)

    # ==========================================================
    # Аугментации (детерминированные)
    # ==========================================================
    transform = A.Compose([
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 10.0), p=1.0),
            A.ISONoise(p=1.0)
        ], p=0.5),
        A.Rotate(limit=10, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8)
    ], additional_targets={"mask": "mask"})

    # ==========================================================
    # Функция аугментации и сохранения (вложена в основную)
    # ==========================================================
    def augment_and_save(image_path, mask_path, out_image_dir, out_mask_dir, filename):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"⚠️ Ошибка чтения: {image_path} / {mask_path}")
            return

        if image.shape[:2] != mask.shape[:2]:
            print(f"⚠️ Несовпадение размеров: {image_path} и {mask_path}")
            return

        base_name = os.path.splitext(filename)[0]

        for i in range(n_aug):
            # Используем собственный seed для каждого augment'а
            local_seed = SEED + i
            random.seed(local_seed)
            np.random.seed(local_seed)

            augmented = transform(image=image, mask=mask)
            aug_image = augmented["image"]
            aug_mask = augmented["mask"]

            cv2.imwrite(os.path.join(out_image_dir, f"{base_name}_aug{i}.jpg"), aug_image)
            cv2.imwrite(os.path.join(out_mask_dir, f"{base_name}_aug{i}.png"), aug_mask)


    # ==========================================================
    # Основной цикл
    # ==========================================================
    for split in SPLITS:
        image_files = [f for f in os.listdir(IMAGE_DIRS[split]) if f.lower().endswith(".jpg")]

        for img_file in tqdm(image_files, desc=f"Аугментация {split}"):
            image_path = os.path.join(IMAGE_DIRS[split], img_file)
            
            mask_file = os.path.splitext(img_file)[0] + "_mask.png"
            mask_path = os.path.join(MASK_DIRS[split], mask_file)

            if not os.path.exists(mask_path):
                print(f"⚠️ Маска не найдена: {mask_path}")
                continue

            augment_and_save(
                image_path,
                mask_path,
                OUTPUT_IMAGE_DIRS[split],
                OUTPUT_MASK_DIRS[split],
                img_file
            )

    print(f"Аугментация завершена. Результаты сохранены в: {BASE_OUTPUT_DIR}")

# Пример вызова функции
if __name__ == "__main__":
    # run_augmentation("PipeSegmentation", n_aug=3, seed=42)
    run_augmentation("PipeBoxSegmentation", n_aug=3, seed=42)
