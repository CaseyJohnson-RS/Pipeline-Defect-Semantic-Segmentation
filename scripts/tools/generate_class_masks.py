import os
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === ПАРАМЕТРЫ (измените под себя) ===
images_dir = "datasets/sorted_images/Obstacle/images"     # Папка с изображениями
labels_dir = "datasets/tmp/train"     # Папка с .txt файлами YOLO
output_dir = "datasets/sorted_images/Obstacle/masks"     # Папка для сохранения масок
target_class = 1                   # Класс, который нужно заливать (целое число)
# =====================================

# Создаём выходную директорию, если её нет
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Поддерживаемые расширения изображений
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

for img_path in Path(images_dir).iterdir():
    if img_path.suffix.lower() not in image_extensions:
        continue

    # Определяем путь к соответствующему файлу разметки
    label_path = Path(labels_dir) / (img_path.stem + ".txt")
    if not label_path.exists():
        print(f"⚠️ Файл разметки не найден для: {img_path.name}")
        continue

    # Читаем изображение, чтобы узнать его размеры
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {img_path}")
        continue

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)  # Чёрная маска

    # Читаем файл разметки
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls != target_class:
                continue

            # Нормализованные координаты YOLO: cx, cy, bw, bh
            cx, cy, bw, bh = map(float, parts[1:5])

            # Переводим в пиксели
            x_center = cx * w
            y_center = cy * h
            box_width = bw * w
            box_height = bh * h

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Ограничиваем координаты размерами изображения
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Заливаем bounding box белым (255)
            mask[y1:y2, x1:x2] = 255

    # Сохраняем маску с тем же расширением, что и у исходного изображения
    mask_filename = img_path.stem + img_path.suffix
    mask_path = Path(output_dir) / mask_filename
    cv2.imwrite(str(mask_path), mask)

    print(f"✅ Создана маска: {mask_path.name}")

print("Готово!")