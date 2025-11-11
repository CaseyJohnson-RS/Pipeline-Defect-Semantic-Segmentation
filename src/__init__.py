from dotenv import load_dotenv
import numpy as np
import random
from pathlib import Path
import torch
import yaml
from typing import Optional

load_dotenv()


def set_seed(SEED):
    """Устанавливает seed для воспроизводимости результатов."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def check_cuda_available():
    """Выводит информацию о доступности CUDA и GPU."""
    print("CUDA доступен:", torch.cuda.is_available())
    print("Число GPU:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Имя GPU:", torch.cuda.get_device_name(0))
        print("Версия CUDA:", torch.version.cuda)
        return True
    return False


def load_yaml_file(filepath: str | Path) -> Optional[dict]:
    """
    Загружает YAML‑файл и возвращает его содержимое в виде словаря.

    Args:
        filepath: путь к YAML‑файлу (строка или объект Path)

    Returns:
        Словарь с данными из YAML‑файла или None при ошибке

    Examples:
        >>> data = load_yaml_file("config.yaml")
        >>> if data is not None:
        ...     print(data)
    """
    file_path = Path(filepath)

    # Проверка существования файла
    if not file_path.exists():
        print(f"Файл не найден: {file_path}")
        return None

    # Проверка, что это файл (а не директория)
    if not file_path.is_file():
        print(f"Указанный путь не является файлом: {file_path}")
        return None

    try:
        with file_path.open('r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            
        # Проверка, что загруженные данные — словарь
        if not isinstance(data, dict):
            print(f"Содержимое файла не является словарём: {file_path}")
            return None
            
        return data

    except yaml.YAMLError as e:
        print(f"Ошибка парсинга YAML в файле {file_path}: {e}")
        return None
    except PermissionError:
        print(f"Нет прав на чтение файла: {file_path}")
        return None
    except UnicodeDecodeError as e:
        print(f"Ошибка декодирования UTF-8 в файле {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка при чтении файла {file_path}: {e}")
        return None

