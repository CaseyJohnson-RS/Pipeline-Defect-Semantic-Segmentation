#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Создание двух отдельных наборов:
  datasets/{NAME}_H{num}      – train
  datasets/{NAME}_H{num}_VAL  – val
Из BASELINE берём:
  BASELINE_DIR/images + BASELINE_DIR/masks           – для train
  BASELINE_DIR_VAL/images + BASELINE_DIR_VAL/masks   – для val
Модель использует ваш SegmentationDataset (OpenCV, без PIL).
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# ------------- CONFIGURATION -------------
DATASET_NAME = input("Enter dataset name: ").strip()
SOURCE_DIR   = Path(f"datasets/{DATASET_NAME}")
BASELINE_DIR      = Path(f"datasets/{DATASET_NAME}_BASELINE")
BASELINE_DIR_VAL  = Path(f"datasets/{DATASET_NAME}_BASELINE_VAL")  # новая папка

ALPHA        = 0.3      # коэффициент для гибридных масок
THRESHOLD    = 0.75     # порог уверенности модели
MODEL_INPUT_SIZE = (512, 702)   # (H, W) – порядок, который ожидает ваш датасет
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED      = 42
TRAIN_VAL_SPLIT  = 0.85
NUM_AUGMENTATIONS = 3
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models import load_model  # noqa: E402

# импортируем ваш датасет
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from src.data import SegmentationDataset  # noqa: E402


# ========== service functions ==========
def parse_polygon(annotation_str):
    if not annotation_str or annotation_str.strip() in {'{}', '""{}""', '""', '"" ""'}:
        return None, None
    cleaned = annotation_str.replace('""', '"')
    if cleaned.strip() == '{}':
        return None, None
    try:
        data = json.loads(cleaned)
        if data.get('name') == 'polygon' and 'all_points_x' in data and 'all_points_y' in data:
            return data['all_points_x'], data['all_points_y']
    except Exception as e:
        print(f"Polygon parsing error: {e}, line: {annotation_str[:50]}...")
    return None, None


def create_mask_from_polygon(shape, pts_x, pts_y):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array(list(zip(pts_x, pts_y)), dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=255)
    return mask


def get_file_mapping(directory: Path):
    """{stem: filename}"""
    return {f.stem: f.name for f in directory.iterdir() if f.is_file()}


def get_pixel_perfect_names(labels_csv: Path):
    df = pd.read_csv(labels_csv)
    df = df[
        (df['region_count'] > 0) &
        (df['region_shape_attributes'].notna()) &
        (~df['region_shape_attributes'].astype(str).str.strip().isin({'{}', '""{}""', ''}))
    ]
    return {Path(f).stem for f in df['filename']}


def predict_mask(model, image_path: Path, target_size, device):
    """
    Возвращает бинарную маску в оригинальном разрешении изображения.
    target_size – (H, W), на который масштабируется картинка перед подачей в модель.
    """
    # 1. читаем оригинал и запоминаем его размер
    orig = cv2.imread(str(image_path))
    if orig is None:
        raise RuntimeError(f"Cannot read image {image_path}")
    h_orig, w_orig = orig.shape[:2]

    # 2. создаём датасет, чтобы переиспользовать его препроцессинг
    ds = SegmentationDataset(
        images_dir=str(image_path.parent),
        masks_dir=str(image_path.parent),  # параметр обязателен, но маска не используется
        img_size=target_size,
        normalize=True
    )
    idx = next((i for i, p in enumerate(ds.images) if Path(p).name == image_path.name), None)
    if idx is None:
        raise FileNotFoundError(f"File {image_path.name} not found in dataset")

    img_tensor, _ = ds[idx]          # CHW, 0-1
    img_tensor = img_tensor.unsqueeze(0).to(device)  # 1CHW

    # 3. предсказание
    with torch.no_grad():
        pred = model.predict(img_tensor)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        # приводим к (H, W)
        if pred.ndim == 4:          # 1,C,H,W
            pred = pred[0, 0]
        elif pred.ndim == 3:        # C,H,W
            pred = pred[0]
        pred = (np.clip(pred, 0, 1) > THRESHOLD)

    # 4. обратно в оригинальное разрешение
    pred_mask = (pred * 255).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return pred_mask


def augment_image_and_mask(image, mask, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    aug_type = random.choice(['hflip', 'vflip', 'rotate'])
    img, msk = image.copy(), mask.copy()

    if aug_type == 'hflip':
        img, msk = cv2.flip(img, 1), cv2.flip(msk, 1)
    elif aug_type == 'vflip':
        img, msk = cv2.flip(img, 0), cv2.flip(msk, 0)
    elif aug_type == 'rotate':
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        msk = cv2.warpAffine(msk, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img, msk, aug_type


# ========== core logic ==========
def build_dataset():
    images_dir = SOURCE_DIR / "images"
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    img_map = get_file_mapping(images_dir)
    print(f"Found {len(img_map)} images")

    perfect = get_pixel_perfect_names(SOURCE_DIR / "labels.csv")
    to_improve = list(set(img_map.keys()) - perfect)
    print(f"{len(to_improve)} files will be improved with model")

    print("Loading model...")
    model = load_model().to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}")

    num = int(input("Enter dataset number: "))

    # две выходные папки
    out_train = Path(f"datasets/{DATASET_NAME}_H{num}")
    out_val   = Path(f"datasets/{DATASET_NAME}_H{num}_VAL")

    # создаём структуру
    (out_train / "images").mkdir(parents=True, exist_ok=True)
    (out_train / "masks").mkdir(parents=True, exist_ok=True)
    (out_val   / "images").mkdir(parents=True, exist_ok=True)
    (out_val   / "masks").mkdir(parents=True, exist_ok=True)

    # разбиваем на train/val
    train_files, val_files = train_test_split(
        to_improve, train_size=TRAIN_VAL_SPLIT, random_state=RANDOM_SEED, shuffle=True
    )

    def process(names, img_root, mask_root):
        """Создаёт гибридные маски и сохраняет картинки"""
        for name in names:
            real_name = img_map[name]
            img_path  = images_dir / real_name
            msk_path  = SOURCE_DIR / "masks" / real_name

            if not img_path.exists() or not msk_path.exists():
                continue

            img = cv2.imread(str(img_path))
            msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue

            try:
                pred = predict_mask(model, img_path, MODEL_INPUT_SIZE, DEVICE)
                pred[msk == 0] = 0          # обнуляем фон
            except Exception as e:
                print(f"Prediction failed for {name}: {e}")
                pred = np.zeros_like(msk)

            hybrid = cv2.addWeighted(msk, ALPHA, pred, 1 - ALPHA, 0)

            cv2.imwrite(str(img_root / real_name), img)
            cv2.imwrite(str(mask_root / f"{Path(real_name).stem}.png"), hybrid)

    # улучшаем маски
    process(train_files, out_train / "images", out_train / "masks")
    process(val_files,   out_val   / "images", out_val   / "masks")

    # копируем baseline
    def copy_baseline(baseline_img_dir, baseline_msk_dir, target_img_root, target_mask_root):
        for img_file in baseline_img_dir.iterdir():
            if not img_file.is_file():
                continue
            mask_name = img_file.stem + ".png"
            bas_mask  = baseline_msk_dir / mask_name

            cv2.imwrite(str(target_img_root / img_file.name), cv2.imread(str(img_file)))
            if bas_mask.exists():
                cv2.imwrite(str(target_mask_root / mask_name),
                            cv2.imread(str(bas_mask), cv2.IMREAD_GRAYSCALE))

    copy_baseline(BASELINE_DIR / "images", BASELINE_DIR / "masks",
                  out_train / "images", out_train / "masks")

    if BASELINE_DIR_VAL.exists():
        copy_baseline(BASELINE_DIR_VAL / "images", BASELINE_DIR_VAL / "masks",
                      out_val / "images", out_val / "masks")
    else:
        print(f"Warning: {BASELINE_DIR_VAL} not found – val baseline skipped")

    # финальная аугментация только train
    def final_aug(img_dir, mask_dir):
        for img_path in img_dir.iterdir():
            if not img_path.is_file() or "_aug" in img_path.name:
                continue
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            img = cv2.imread(str(img_path))
            msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            for i in range(NUM_AUGMENTATIONS):
                a_img, a_msk, _ = augment_image_and_mask(
                    img, msk, seed=RANDOM_SEED + hash(img_path.stem) % 1000 + i)
                cv2.imwrite(str(img_dir / f"{img_path.stem}_aug{i}{img_path.suffix}"), a_img)
                cv2.imwrite(str(mask_dir / f"{img_path.stem}_aug{i}.png"), a_msk)

    final_aug(out_train / "images", out_train / "masks")

    # статистика
    def cnt(p): return len([x for x in p.iterdir() if x.is_file()])
    print("\n✅ Done!")
    print(f"   Train dataset : {out_train}  – {cnt(out_train / 'images')} imgs")
    print(f"   Val dataset   : {out_val}    – {cnt(out_val   / 'images')}   imgs")


if __name__ == "__main__":
    build_dataset()