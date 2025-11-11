import os
import cv2
import numpy as np
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# пути к папкам
folder1 = r'datasets/PipeSegmentation/images/train/'
folder2 = r'datasets/tmp/'
folder3 = r'datasets/PipeSegmentation/masks/train/'

os.makedirs(folder3, exist_ok=True)

for img_name in os.listdir(folder1):
    img_path = os.path.join(folder1, img_name)
    name_wo_ext = os.path.splitext(img_name)[0]
    mask_path = os.path.join(folder2, name_wo_ext + '.txt')

    if not os.path.exists(mask_path):
        print(f'Нет маски для {img_name}')
        continue

    # читаем изображение, чтобы знать размер
    image = cv2.imread(img_path)
    if image is None:
        print(f'Не удалось прочитать {img_name}')
        continue
    h, w = image.shape[:2]

    # создаём пустую маску (чёрный фон)
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(mask_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        coords = list(map(float, parts[1:]))

        # координаты: x y x y ... (нормализованные)
        points = np.array([
            [coords[i] * w, coords[i + 1] * h]
            for i in range(0, len(coords), 2)
        ], np.int32)

        # рисуем заполненную область на маске
        cv2.fillPoly(mask, [points], 255)

    out_path = os.path.join(folder3, name_wo_ext + '_mask.png')
    cv2.imwrite(out_path, mask)
    print(f'Маска сохранена: {out_path}')
