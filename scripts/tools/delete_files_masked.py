import os
import re
from pathlib import Path
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def delete_non_matching_files(directory_path):
    """
    Удаляет файлы, имя которых НЕ соответствует маске:
    <6 цифр>.<расширение> (например, 123456.txt)


    :param directory_path: путь к директории для обработки
    """
    # Регулярное выражение: ровно 6 цифр в начале, затем точка и расширение
    pattern = re.compile(r'^\d{6}\.[^.]+$')

    dir_path = Path(directory_path)

    if not dir_path.exists():
        print(f"Ошибка: директория не существует: {directory_path}")
        return

    if not dir_path.is_dir():
        print(f"Ошибка: путь не является директорией: {directory_path}")
        return

    deleted_count = 0
    for item in dir_path.iterdir():
        if item.is_file():
            if not pattern.match(item.name):  # Если имя НЕ подходит под маску
                try:
                    item.unlink()
                    print(f"Удалено: {item.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Ошибка при удалении {item.name}: {e}")

    print(f"\nВсего удалено файлов: {deleted_count}")

# --- Основной код ---
if __name__ == "__main__":
    # Запрос пути у пользователя
    target_directory = input("Введите путь к директории: ").strip()

    if not target_directory:
        print("Путь не указан!")
    else:
        delete_non_matching_files(target_directory)
