import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --- Настройки --- #
images_dir = "datasets/OriginalDS/images/images/train"       # Путь к изображениям
labels_dir = "datasets/OriginalDS/labels/labels/train"       # Путь к аннотациям (YOLO .txt)
output_dir = "sorted_images"      # Папка для отсортированных изображений

# Классы из вашего config.yaml
class_names = {
    0: "Deformation",
    1: "Obstacle",
    2: "Rupture",
    3: "Disconnect",
    4: "Misalignment",
    5: "Deposition"
}

# --- Создание папок для каждого класса --- #
output_path = Path(output_dir)
output_path.mkdir(exist_ok=True)

for class_id, class_name in class_names.items():
    (output_path / class_name).mkdir(exist_ok=True)

# --- Сопоставление изображений и аннотаций --- #
images_path = Path(images_dir)
labels_path = Path(labels_dir)

# Предполагаем, что аннотации имеют тот же basename, что и изображения, но с расширением .txt
for label_file in labels_path.glob("*.txt"):
    image_stem = label_file.stem  # Без расширения

    # Поддерживаемые расширения изображений (можно расширить)
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    image_file = None
    for ext in supported_extensions:
        candidate = images_path / (image_stem + ext)
        if candidate.exists():
            image_file = candidate
            break

    if not image_file:
        print(f"⚠️  Предупреждение: не найдено изображение для аннотации {label_file}")
        continue

    # Чтение аннотаций
    with open(label_file, 'r') as f:
        lines = f.readlines()

    found_classes = set()
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            if class_id in class_names:
                found_classes.add(class_id)
        except ValueError:
            continue  # Игнорируем некорректные строки

    # Копируем изображение в каждую папку, соответствующую найденным классам
    for class_id in found_classes:
        dest_dir = output_path / class_names[class_id]
        dest_file = dest_dir / image_file.name
        shutil.copy2(image_file, dest_file)

print("✅ Сортировка завершена!")