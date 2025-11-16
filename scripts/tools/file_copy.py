import os
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# пути к папкам
folder1 = r'datasets/PipeBoxSegmentation/masks/train'
folder2 = r'datasets/sorted_images/Disconnect/masks'
folder3 = r'datasets/sorted_images/Disconnect/images'

# создаём папку назначения, если вдруг её нет
os.makedirs(folder2, exist_ok=True)

# получаем множество имён без расширений из папки 3
files_in_folder3 = {
    os.path.splitext(f)[0] for f in os.listdir(folder3)
    if os.path.isfile(os.path.join(folder3, f))
}

# перебираем файлы в папке 1
for filename in os.listdir(folder1):
    source_path = os.path.join(folder1, filename)
    name_without_ext = os.path.splitext(filename)[0]
    destination_path = os.path.join(folder2, filename)

    # проверяем наличие имени без расширения в папке 3
    if name_without_ext in files_in_folder3:
        shutil.copy2(source_path, destination_path)  # copy2 сохраняет метаданные
        print(f'Скопирован: {filename}')
    else:
        print(f'Пропущен: {filename}')
