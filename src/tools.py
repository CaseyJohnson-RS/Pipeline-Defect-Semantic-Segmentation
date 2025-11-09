from dotenv import load_dotenv
import numpy as np
import random
import torch
import os

load_dotenv()


MODELS_DIRECTORY = os.getenv('MODELS_DIRECTORY')


def check_cuda_available():
    """Выводит информацию о доступности CUDA и GPU."""
    print("CUDA доступен:", torch.cuda.is_available())
    print("Число GPU:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Имя GPU:", torch.cuda.get_device_name(0))
        print("Версия CUDA:", torch.version.cuda)


def set_seed(SEED):
    """Устанавливает seed для воспроизводимости результатов."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def save_model(model, model_name="model"):
    """
    Сохраняет модель PyTorch в папку 'MODELS_DIRECTORY' с датой и временем в имени.
    
    Args:
        model: экземпляр модели PyTorch
        model_name (str): базовое имя файла (без расширения)
    
    Returns:
        str: полный путь к сохранённому файлу
    """

    # 1. Создаём папку models, если её нет
    os.makedirs(MODELS_DIRECTORY, exist_ok=True)
    
    filename = f"{model_name}.pth"
    save_path = os.path.join(MODELS_DIRECTORY, filename)
    
    # 4. Сохраняем state_dict модели
    torch.save(model.state_dict(), save_path)
    
    print(f"Model saved localy: {save_path}")
    return save_path


def check_dataset_dirs(dataset_pash: str) -> bool:

    for data_dir in ['images', 'masks']:
        for divide_dir in ['train', 'val']:
            if not os.path.isdir(os.path.join(dataset_pash, data_dir, divide_dir)):
                return False
    return True 


def gradient_color(value, min_val=0, max_val=1, reverse=False):
    """
    Плавный градиент для числового значения.
    
    Args:
        value: значение для окрашивания
        min_val: минимальное ожидаемое значение (по умолчанию 0)
        max_val: максимальное ожидаемое значение (по умолчанию 1)
        reverse: если True — градиент от зелёного к красному; 
                  если False — от красного к зелёному (стандарт)
    
    Returns:
        Строка с ANSI‑кодом цвета и форматированным значением
    """
    # Нормализуем value в диапазон [0, 1]
    v = (value - min_val) / (max_val - min_val)
    v = max(0, min(1, v))  # ограничиваем в пределах [0, 1]

    if reverse:
        # Градиент: зелёный (v=0) → красный (v=1)
        r = int(255 * v)      # r растёт от 0 до 255
        g = int(255 * (1 - v)) # g падает от 255 до 0
        b = 0
    else:
        # Градиент: красный (v=0) → зелёный (v=1)
        r = int(255 * (1 - v)) # r падает от 255 до 0
        g = int(255 * v)       # g растёт от 0 до 255
        b = 0

    return f"\033[38;2;{r};{g};{b}m{value:.3f}\033[0m"

if __name__ == '__main__':
    print(f"IoU (нормальный): {gradient_color(0.3, 0, 1, reverse=False)}")    # красноватый
    print(f"Dice (нормальный): {gradient_color(0.85, 0, 1, reverse=False)}")  # зеленоватый

    print(f"Error (инверсный): {gradient_color(0.3, 0, 1, reverse=True)}")   # зеленоватый (низкая ошибка — хорошо)
    print(f"Loss (инверсный): {gradient_color(0.85, 0, 1, reverse=True)}") # красноватый (высокая потеря — плохо)
