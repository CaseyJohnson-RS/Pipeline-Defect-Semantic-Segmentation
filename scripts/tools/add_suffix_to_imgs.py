import os
import sys
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def add_suffix_to_images(directory, suffix="_mask", image_extensions=None):
    """
    Добавляет суффикс к именам всех изображений в указанной директории.
    
    Args:
        directory (str): Путь к директории с изображениями
        suffix (str): Суффикс для добавления (по умолчанию: "_mask")
        image_extensions (list): Список расширений для обработки.
                                По умолчанию: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    Returns:
        dict: Статистика операции {'processed': int, 'errors': int, 'skipped': int}
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    directory = Path(directory)
    stats = {'processed': 0, 'errors': 0, 'skipped': 0}
    
    if not directory.exists():
        print(f"Ошибка: Директория '{directory}' не существует!")
        return stats
    
    if not directory.is_dir():
        print(f"Ошибка: '{directory}' не является директорией!")
        return stats
    
    print(f"Обработка директории: {directory.absolute()}")
    print(f"Добавляемый суффикс: '{suffix}'")
    print("-" * 50)
    
    for file_path in directory.iterdir():
        # Проверяем, что это файл
        if not file_path.is_file():
            continue
        
        # Проверяем расширение файла
        if file_path.suffix.lower() not in image_extensions:
            print(f"Пропущен: {file_path.name} (неподдерживаемое расширение)")
            stats['skipped'] += 1
            continue
        
        # Проверяем, не содержит ли имя уже суффикс
        if suffix in file_path.stem:
            print(f"Пропущен: {file_path.name} (суффикс уже существует)")
            stats['skipped'] += 1
            continue
        
        # Формируем новое имя
        new_name = f"{file_path.stem}{suffix}{file_path.suffix}"
        new_path = file_path.with_name(new_name)
        
        # Проверяем, не существует ли уже файл с новым именем
        if new_path.exists():
            print(f"Ошибка: {file_path.name} -> {new_name} (файл уже существует)")
            stats['errors'] += 1
            continue
        
        try:
            # Переименовываем файл
            file_path.rename(new_path)
            print(f"Успешно: {file_path.name} -> {new_name}")
            stats['processed'] += 1
        except Exception as e:
            print(f"Ошибка: {file_path.name} -> {new_name} ({e})")
            stats['errors'] += 1
    
    # Вывод итогов
    print("-" * 50)
    print(f"Обработано файлов: {stats['processed']}")
    print(f"Пропущено файлов: {stats['skipped']}")
    print(f"Ошибок: {stats['errors']}")
    
    return stats

# Пример использования
if __name__ == "__main__":
    # Вариант 1: Параметры в коде
    # folder = "path/to/your/images"
    # add_suffix_to_images(folder)
    
    # Вариант 2: Интерактивный ввод
    folder = input("Введите путь к директории с изображениями: ").strip()
    if folder:
        add_suffix_to_images(folder)
    else:
        print("Ошибка: путь не указан!")
    
    # Вариант 3: Обработка поддиректорий
    # import sys
    # if len(sys.argv) > 1:
    #     folder = sys.argv[1]
    #     add_suffix_to_images(folder)