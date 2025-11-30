import os
import json
import pandas as pd
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============== CONFIGURATION ==============
DATASET_NAME = input("Enter dataset name: ")
SOURCE_DIR = Path(f"datasets/{DATASET_NAME}")
BASELINE_DIR = Path(f"datasets/{DATASET_NAME}_BASELINE")
EVAL_DIR = Path(f"datasets/{DATASET_NAME}_EVAL")
VAL_DIR = Path(f"datasets/{DATASET_NAME}_BASELINE_VAL")  # <-- NEW

RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.85  # 85% for train
NUM_AUGMENTATIONS = 3   # Number of augmented images per original
# ============================================

def parse_polygon(annotation_str):
    """Parses JSON string with polygon and returns coordinates"""
    try:
        if not annotation_str or annotation_str.strip() in ['{}', '""{}""', '""']:
            return None, None
        cleaned = annotation_str.replace('""', '"')
        if not cleaned or cleaned.strip() == '{}':
            return None, None
        data = json.loads(cleaned)
        if data.get('name') == 'polygon' and 'all_points_x' in data and 'all_points_y' in data:
            return data['all_points_x'], data['all_points_y']
        return None, None
    except Exception as e:
        print(f"Polygon parsing error: {e}, line: {annotation_str[:50]}...")
        return None, None

def create_mask_from_polygon(image_shape, points_x, points_y):
    """Creates binary mask from polygon coordinates"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array(list(zip(points_x, points_y)), dtype=np.int32)
    cv2.fillPoly(mask, [points], color=255)
    return mask

def copy_or_create_dirs(*dirs):
    """Creates directories if they don't exist"""
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_file_mapping(directory):
    """Создает словарь {имя_без_расширения: полное_имя_файла}"""
    mapping = {}
    for file_path in directory.iterdir():
        if file_path.is_file():
            name_without_ext = file_path.stem
            mapping[name_without_ext] = file_path.name
    return mapping

def augment_image_and_mask(image, mask, seed=None):
    """Применяет случайную лёгкую аугментацию к изображению и маске"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    augmented_image = image.copy()
    augmented_mask = mask.copy()
    aug_type = random.choice(['hflip', 'vflip', 'rotate'])
    if aug_type == 'hflip':
        augmented_image = cv2.flip(augmented_image, 1)
        augmented_mask = cv2.flip(augmented_mask, 1)
    elif aug_type == 'vflip':
        augmented_image = cv2.flip(augmented_image, 0)
        augmented_mask = cv2.flip(augmented_mask, 0)
    elif aug_type == 'rotate':
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented_image = cv2.warpAffine(augmented_image, M, (w, h))
        augmented_mask = cv2.warpAffine(augmented_mask, M, (w, h))
    return augmented_image, augmented_mask, aug_type

def process_files(name_list, img_dest_dir, mask_dest_dir, split_name="", apply_aug=False):
    """Обрабатывает список имен файлов (без расширения)"""
    print(f"\nОбработка {split_name}: {len(name_list)} файлов" + 
          (" + аугментация" if apply_aug else ""))
    processed_count = 0
    for name in name_list:
        real_filename = image_mapping[name]
        src_image_path = images_dir / real_filename
        try:
            image = cv2.imread(str(src_image_path))
            if image is None:
                print(f"Error reading image {src_image_path}")
                continue
        except Exception as e:
            print(f"Error reading {src_image_path}: {e}")
            continue
        if name not in annotations_by_name:
            print(f"Warning: No annotations for file {name}")
            continue
        annotations = annotations_by_name[name]
        if not annotations:
            print(f"Warning: Empty annotations for file {name}")
            continue
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        valid_polygons_found = False
        for row in annotations:
            points_x, points_y = parse_polygon(row['region_shape_attributes'])
            if points_x is not None and points_y is not None:
                mask = create_mask_from_polygon(image.shape, points_x, points_y)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
                valid_polygons_found = True
        if not valid_polygons_found:
            print(f"Warning: No valid polygons found for file {name}")
            continue
        dest_image_path = img_dest_dir / real_filename
        mask_filename = Path(real_filename).stem + ".png"
        dest_mask_path = mask_dest_dir / mask_filename
        try:
            cv2.imwrite(str(dest_image_path), image)
            cv2.imwrite(str(dest_mask_path), combined_mask)
            processed_count += 1
        except Exception as e:
            print(f"Ошибка сохранения {real_filename}: {e}")
            continue
        if apply_aug:
            for i in range(NUM_AUGMENTATIONS):
                aug_seed = RANDOM_SEED + hash(name) % 1000 + i
                aug_image, aug_mask, aug_type = augment_image_and_mask(
                    image, combined_mask, seed=aug_seed
                )
                aug_image_name = f"{Path(real_filename).stem}_aug{i}{Path(real_filename).suffix}"
                aug_mask_name = f"{Path(real_filename).stem}_aug{i}.png"
                aug_image_path = img_dest_dir / aug_image_name
                aug_mask_path = mask_dest_dir / aug_mask_name
                cv2.imwrite(str(aug_image_path), aug_image)
                cv2.imwrite(str(aug_mask_path), aug_mask)
                processed_count += 1
                if len(name_list) <= 3:
                    print(f"    Оригинал: {real_filename} -> {aug_type} -> {aug_image_name}")
    print(f"  Сохранено {processed_count} файлов (включая аугментированные)")

def main():
    global images_dir, image_mapping, annotations_by_name
    images_dir = SOURCE_DIR / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Папка {images_dir} не найдена!")
    image_mapping = get_file_mapping(images_dir)
    print(f"Найдено {len(image_mapping)} файлов в папке images")
    labels_path = SOURCE_DIR / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Файл {labels_path} не найден!")
    df = pd.read_csv(labels_path)
    initial_count = len(df)
    df = df[(df['region_count'] > 0) & 
            (df['region_shape_attributes'].notna()) & 
            (df['region_shape_attributes'].str.strip() != '{}') &
            (df['region_shape_attributes'].str.strip() != '""{}""')]
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"✅ Фильтровано {filtered_count} файлов без разметки (region_count=0)")
    annotations_by_name = {}
    for _, row in df.iterrows():
        csv_filename = row['filename']
        name_without_ext = Path(csv_filename).stem
        if name_without_ext in image_mapping:
            if name_without_ext not in annotations_by_name:
                annotations_by_name[name_without_ext] = []
            annotations_by_name[name_without_ext].append(row)
        else:
            print(f"Предупреждение: Файл '{csv_filename}' из CSV не найден в папке images")
    valid_names = list(annotations_by_name.keys())
    total_files = len(valid_names)
    print(f"\nНайдено {total_files} файлов с pixel-perfect масками и соответствующими изображениями")
    if total_files == 0:
        print("Ошибка: Не найдено ни одного файла для обработки. Проверьте соответствие имен в CSV и папке images.")
        return
    baseline_names, eval_names = train_test_split(
        valid_names,
        test_size=0.5,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    train_names, val_names = train_test_split(
        baseline_names,
        train_size=TRAIN_VAL_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    # === NEW: Create directories ===
    baseline_images = BASELINE_DIR / "images"
    baseline_masks = BASELINE_DIR / "masks"
    val_images = VAL_DIR / "images"
    val_masks = VAL_DIR / "masks"
    eval_images = EVAL_DIR / "images"
    eval_masks = EVAL_DIR / "masks"
    copy_or_create_dirs(baseline_images, baseline_masks,
                        val_images, val_masks,
                        eval_images, eval_masks)
    # === Process files ===
    process_files(train_names, baseline_images, baseline_masks, "BASELINE (train)", apply_aug=True)
    process_files(val_names, val_images, val_masks, "VAL", apply_aug=False)
    process_files(eval_names, eval_images, eval_masks, "EVAL", apply_aug=False)
    print("\n✅ Готово! Датасеты созданы:")
    print(f"   - BASELINE: {BASELINE_DIR}")
    print(f"   - VAL: {VAL_DIR}")
    print(f"   - EVAL: {EVAL_DIR}")

if __name__ == "__main__":
    main()