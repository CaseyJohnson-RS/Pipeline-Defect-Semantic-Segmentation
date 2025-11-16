import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import torch
from torch.utils.data import Dataset
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.models.factories import load_unet_attention  # noqa: E402

# ============== CONFIGURATION ==============
DATASET_NAME = input("Enter dataset name: ")
SOURCE_DIR = Path(f"datasets/{DATASET_NAME}")
BASELINE_DIR = Path(f"datasets/{DATASET_NAME}_BASELINE")
OUTPUT_DIR = Path(f"datasets/{DATASET_NAME}_H")  # Будет дополнен значением alpha

# Параметры модели и улучшения масок
ALPHA = 0.7  # Коэффициент для гибридных масок
MODEL_INPUT_SIZE = (702, 512)  # Размер входа модели
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Параметры аугментации
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.85  # 85% для train
NUM_AUGMENTATIONS = 3   # Количество аугментированных изображений на одно оригинальное
# ============================================

class ImageMaskDataset(Dataset):
    """Датасет для загрузки изображений и масок"""
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask, self.image_paths[idx]

def parse_polygon(annotation_str):
    """Парсит JSON-строку с полигоном и возвращает координаты"""
    try:
        cleaned = annotation_str.replace('""', '"')
        data = json.loads(cleaned)
        if data['name'] == 'polygon':
            return data['all_points_x'], data['all_points_y']
        return None, None
    except Exception as e:
        print(f"Ошибка парсинга полигона: {e}")
        return None, None

def create_mask_from_polygon(image_shape, points_x, points_y):
    """Создает бинарную маску из координат полигона"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array(list(zip(points_x, points_y)), dtype=np.int32)
    cv2.fillPoly(mask, [points], color=255)
    return mask

def copy_or_create_dirs(*dirs):
    """Создает директории, если они не существуют"""
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

def get_pixel_perfect_names(labels_path):
    """Получает список имен файлов с pixel-perfect масками"""
    df = pd.read_csv(labels_path)
    pixel_perfect_names = set()
    for filename in df['filename']:
        pixel_perfect_names.add(Path(filename).stem)
    return pixel_perfect_names

def preprocess_for_model(image, target_size):
    """Подготавливает изображение для модели"""
    # Resize
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    # Normalize
    image_norm = image_resized.astype(np.float32) / 255.0
    # Convert to tensor format: (C, H, W)
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)
    return image_tensor

def predict_mask(model, image_path, target_size, device):
    """Получает предсказание маски от модели"""
    # Загрузка и предобработка изображения
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    # Подготовка для модели
    image_tensor = preprocess_for_model(image, target_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Добавляем batch dimension
    
    # Предсказание
    with torch.no_grad():
        prediction = model.predict(image_tensor)
        
        # === ИСПРАВЛЕНИЕ: Конвертация тензора в NumPy ===
        # Обработка различных форматов выхода модели
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu()
        
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.numpy()
        # =================================================
        
        # Теперь prediction - это numpy array
        # Предполагаем, что prediction имеет форму (1, H, W) или (1, 1, H, W) или (H, W)
        if prediction.ndim == 4:
            # (B, C, H, W) -> берем первый канал первого батча
            prediction = prediction[0, 0]
        elif prediction.ndim == 3:
            # (B, H, W) -> берем первый батч
            prediction = prediction[0]
        # Если ndim == 2, то это уже (H, W)
        
        # Убеждаемся, что значения в диапазоне [0, 1]
        prediction = np.clip(prediction, 0, 1)
        
    # Resize обратно к оригинальному размеру
    pred_mask = (prediction * 255).astype(np.uint8)
    pred_mask_resized = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
    return pred_mask_resized

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

def main():
    global images_dir, image_mapping, annotations_by_name
    
    # 1. Сканируем папку images
    images_dir = SOURCE_DIR / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Папка {images_dir} не найдена!")
    
    image_mapping = get_file_mapping(images_dir)
    print(f"Найдено {len(image_mapping)} файлов в папке images")
    
    # 2. Получаем список файлов с pixel-perfect масками
    labels_path = SOURCE_DIR / "labels.csv"
    pixel_perfect_names = get_pixel_perfect_names(labels_path)
    print(f"Найдено {len(pixel_perfect_names)} файлов с pixel-perfect масками")
    
    # 3. Определяем файлы без pixel-perfect масок (для улучшения)
    all_names = set(image_mapping.keys())
    names_to_improve = list(all_names - pixel_perfect_names)
    print(f"Найдено {len(names_to_improve)} файлов для улучшения моделью")
    
    # 4. Загрузка модели
    print("\nЗагрузка модели...")
    model = load_unet_attention()
    model.to(DEVICE)
    model.eval()
    print(f"Модель загружена на {DEVICE}")

    dataset_num = int(input("Enter dataset number: "))

    # 5. Создание выходной директории с учетом alpha
    output_dir = Path(f"./datasets/{DATASET_NAME}_H{dataset_num}")
    output_images_train = output_dir / "images" / "train"
    output_images_val = output_dir / "images" / "val"
    output_masks_train = output_dir / "masks" / "train"
    output_masks_val = output_dir / "masks" / "val"
    
    copy_or_create_dirs(output_images_train, output_images_val, 
                       output_masks_train, output_masks_val)
    
    # 6. Обработка файлов для улучшения масок
    train_files, val_files = train_test_split(
        names_to_improve,
        train_size=TRAIN_VAL_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"\nОбработка {len(train_files)} train файлов и {len(val_files)} val файлов...")
    
    def process_and_save(name_list, img_dest, mask_dest, split_name):
        """Обрабатывает файлы, создает гибридные маски и сохраняет"""
        print(f"\n{split_name}: создание гибридных масок...")
        
        for name in name_list:
            real_filename = image_mapping[name]
            src_image_path = images_dir / real_filename
            src_mask_path = SOURCE_DIR / "masks" / real_filename
            
            # Проверка существования файлов
            if not src_image_path.exists():
                print(f"Предупреждение: Изображение {src_image_path} не найдено")
                continue
            if not src_mask_path.exists():
                print(f"Предупреждение: Маска {src_mask_path} не найдена")
                continue
            
            # Загрузка изображения и оригинальной маски
            image = cv2.imread(str(src_image_path))
            original_mask = cv2.imread(str(src_mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or original_mask is None:
                print(f"Ошибка чтения файлов для {name}")
                continue
            
            # Получение предсказания модели
            try:
                predicted_mask = predict_mask(model, src_image_path, MODEL_INPUT_SIZE, DEVICE)
            except Exception as e:
                print(f"Ошибка предсказания для {name}: {e}")
                # Если ошибка, используем только оригинальную маску
                predicted_mask = np.zeros_like(original_mask)
            
            # Создание гибридной маски
            # original_mask * alpha + predicted_mask * (1 - alpha)
            hybrid_mask = cv2.addWeighted(
                original_mask, ALPHA,
                predicted_mask, (1 - ALPHA),
                0
            )
            
            # Сохранение изображения и гибридной маски
            dest_image_path = img_dest / real_filename
            mask_filename = Path(real_filename).stem + ".png"
            dest_mask_path = mask_dest / mask_filename
            
            try:
                cv2.imwrite(str(dest_image_path), image)
                cv2.imwrite(str(dest_mask_path), hybrid_mask)
            except Exception as e:
                print(f"Ошибка сохранения {name}: {e}")
                continue
            
            if len(name_list) <= 5:
                print(f"  {name}: гибридная маска создан (alpha={ALPHA})")
    
    # Создание гибридных масок
    process_and_save(train_files, output_images_train, output_masks_train, "Hybrid train")
    process_and_save(val_files, output_images_val, output_masks_val, "Hybrid val")
    
    # 7. Добавление НЕ аугментированных изображений из BASELINE
    print("\nДобавление оригиналов из BASELINE...")
    
    baseline_train_dir = BASELINE_DIR / "images" / "train"
    baseline_train_files = [f for f in baseline_train_dir.iterdir() 
                           if f.is_file() and "_aug" not in f.name]
    
    for image_file in baseline_train_files:
        # Копируем изображение
        src_image = image_file
        dest_image = output_images_train / image_file.name
        if not dest_image.exists():
            image = cv2.imread(str(src_image))
            cv2.imwrite(str(dest_image), image)
        
        # Копируем маску
        mask_name = image_file.stem + ".png"
        src_mask = BASELINE_DIR / "masks" / "train" / mask_name
        dest_mask = output_masks_train / mask_name
        
        if src_mask.exists() and not dest_mask.exists():
            mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(dest_mask), mask)
    
    # Добавляем валидационные файлы из BASELINE
    baseline_val_dir = BASELINE_DIR / "images" / "val"
    baseline_val_files = [f for f in baseline_val_dir.iterdir() if f.is_file()]
    
    for image_file in baseline_val_files:
        # Копируем изображение
        src_image = image_file
        dest_image = output_images_val / image_file.name
        if not dest_image.exists():
            image = cv2.imread(str(src_image))
            cv2.imwrite(str(dest_image), image)
        
        # Копируем маску
        mask_name = image_file.stem + ".png"
        src_mask = BASELINE_DIR / "masks" / "val" / mask_name
        dest_mask = output_masks_val / mask_name
        
        if src_mask.exists() and not dest_mask.exists():
            mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(dest_mask), mask)
    
    print(f"Добавлено {len(baseline_train_files)} train и {len(baseline_val_files)} val файлов из BASELINE")
    
    # 8. Аугментация объединённого датасета (только train)
    def apply_final_augmentation(img_dir, mask_dir):
        """Применяет аугментацию ко всем файлам в директории"""
        print(f"\nПрименение финальной аугментации к {img_dir}...")
        
        image_files = [f for f in img_dir.iterdir() if f.is_file() and "_aug" not in f.name]
        
        for image_file in image_files:
            image = cv2.imread(str(image_file))
            mask_file = mask_dir / (image_file.stem + ".png")
            
            if not mask_file.exists():
                continue
            
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            for i in range(NUM_AUGMENTATIONS):
                aug_seed = RANDOM_SEED + hash(image_file.stem) % 1000 + i
                
                aug_image, aug_mask, aug_type = augment_image_and_mask(
                    image, mask, seed=aug_seed
                )
                
                # Сохраняем аугментированные версии
                aug_image_name = f"{image_file.stem}_aug{i}{image_file.suffix}"
                aug_mask_name = f"{image_file.stem}_aug{i}.png"
                
                cv2.imwrite(str(img_dir / aug_image_name), aug_image)
                cv2.imwrite(str(mask_dir / aug_mask_name), aug_mask)
    
    apply_final_augmentation(output_images_train, output_masks_train)
    
    # 9. Вывод статистики
    print("\n✅ Готово! Датасет создан:")
    print(f"   - Путь: {output_dir}")
    print(f"   - Alpha: {ALPHA}")
    
    train_images = list(output_images_train.iterdir())
    val_images = list(output_images_val.iterdir())
    total_images = len(train_images) + len(val_images)
    
    print("\n📊 Статистика:")
    print(f"   - Train: {len(train_images)} изображений")
    print(f"   - Val:   {len(val_images)} изображений")
    print(f"   - Total: {total_images} изображений")

if __name__ == "__main__":
    main()