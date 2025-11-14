import os
import shutil
import random
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Пути
images_train = r'datasets/Deformation/images/train'
masks_train = r'datasets/Deformation/masks/train'

images_val = r'datasets/Deformation/images/val'
masks_val = r'datasets/Deformation/masks/val'

# Создаём папки для валидации, если их нет
os.makedirs(images_val, exist_ok=True)
os.makedirs(masks_val, exist_ok=True)

# Получаем список всех изображений
image_files = [f for f in os.listdir(images_train) if os.path.isfile(os.path.join(images_train, f))]

# Перемешиваем
random.shuffle(image_files)

# Берём 20%
val_count = int(len(image_files) * 0.2)
val_files = image_files[:val_count]

for img_name in val_files:
    name_wo_ext = os.path.splitext(img_name)[0]

    img_src = os.path.join(images_train, img_name)
    img_dst = os.path.join(images_val, img_name)

    # ищем соответствующую маску (начинается с имени картинки)
    mask_candidates = [m for m in os.listdir(masks_train) if m.startswith(name_wo_ext)]

    # копируем картинку
    shutil.move(img_src, img_dst)

    # копируем маску(и), если есть
    for mask_name in mask_candidates:
        mask_src = os.path.join(masks_train, mask_name)
        mask_dst = os.path.join(masks_val, mask_name)
        shutil.move(mask_src, mask_dst)

print(f'Разделение завершено: {val_count} изображений перемещено в val.')
