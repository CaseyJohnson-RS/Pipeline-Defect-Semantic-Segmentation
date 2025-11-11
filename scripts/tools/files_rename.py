import os
import re
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def rename_files_in_directory(directory_path):
    # Проверяем, существует ли директория
    if not os.path.isdir(directory_path):
        print(f"Ошибка: директория '{directory_path}' не существует или не является директорией.")
        return

    # Регулярное выражение: 6 цифр, затем подчёркивание, затем что-то, затем .txt
    pattern = re.compile(r'^(\d{6})_.*\.txt$')

    for filename in os.listdir(directory_path):
        match = pattern.match(filename)
        if match:
            # Получаем 6-значное число
            six_digits = match.group(1)
            new_filename = f"{six_digits}.txt"
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"Переименовано: {filename} → {new_filename}")
            except Exception as e:
                print(f"Ошибка при переименовании {filename}: {e}")

# Пример использования
if __name__ == "__main__":
    # Укажите путь к директории
    target_directory = "datasets/tmp/"  # ← ИЗМЕНИТЕ ЭТОТ ПУТЬ!
    rename_files_in_directory(target_directory)
