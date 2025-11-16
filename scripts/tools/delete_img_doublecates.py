import os
import hashlib
from PIL import Image
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def image_hash(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        data = img.tobytes()
        return hashlib.md5(data).hexdigest()

def remove_duplicates(directory):
    seen = {}
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)

        if not os.path.isfile(full_path):
            continue

        try:
            file_hash = image_hash(full_path)
        except Exception:
            continue  # пропускаем файлы, которые не открываются как картинки

        if file_hash in seen:
            os.remove(full_path)
        else:
            seen[file_hash] = full_path

# Пример вызова
remove_duplicates("datasets/sorted_images/Obstacle")
