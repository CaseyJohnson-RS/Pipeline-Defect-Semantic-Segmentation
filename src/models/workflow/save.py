import torch
import os

MODELS_DIR = os.getenv("MODELS_DIR")


def save_model(model, model_name="model"):
    """
    Сохраняет модель PyTorch в папку 'MODELS_DIR' с датой и временем в имени.

    Args:
        model: экземпляр модели PyTorch
        model_name (str): базовое имя файла (без расширения)

    Returns:
        str: полный путь к сохранённому файлу
    """

    os.makedirs(MODELS_DIR, exist_ok=True)

    filename = f"{model_name}.pth"
    save_path = os.path.join(MODELS_DIR, filename)

    torch.save(model, save_path)

    print(f"Model saved localy: {save_path}")
    return save_path
