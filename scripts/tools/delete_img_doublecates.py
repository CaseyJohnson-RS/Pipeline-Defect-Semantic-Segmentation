import os
import imagehash
from PIL import Image
from pathlib import Path
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def remove_duplicate_images(directory, hash_size=8, delete_duplicates=True):
    """
    Удаляет дубликаты изображений в указанной директории.

    :param directory: Путь к папке с изображениями.
    :param hash_size: Размер хеша (меньше — быстрее, но менее точно).
    :param delete_duplicates: Если True — удаляет дубликаты, иначе только выводит их.
    """
    image_hashes = {}
    duplicates = []

    # Поддерживаемые форматы изображений
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    for file_path in Path(directory).iterdir():
        if file_path.suffix.lower() not in supported_formats:
            continue

        try:
            with Image.open(file_path) as img:
                # Преобразуем в RGB, если необходимо (например, для GIF или RGBA)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                hash_val = imagehash.average_hash(img, hash_size=hash_size)
        except Exception as e:
            print(f"Не удалось обработать {file_path}: {e}")
            continue

        if hash_val in image_hashes:
            duplicates.append(str(file_path))
            if delete_duplicates:
                os.remove(file_path)
                print(f"Удалён дубликат: {file_path}")
        else:
            image_hashes[hash_val] = str(file_path)

    if not delete_duplicates:
        print("Найдены дубликаты:")
        for dup in duplicates:
            print(dup)

    print(f"Всего найдено и {'удалено' if delete_duplicates else 'обнаружено'} дубликатов: {len(duplicates)}")

# Пример использования:
if __name__ == "__main__":
    folder_path = input("Введите путь к папке с изображениями: ").strip()
    if os.path.isdir(folder_path):
        remove_duplicate_images(folder_path, delete_duplicates=True)
    else:
        print("Указанный путь не является директорией.")