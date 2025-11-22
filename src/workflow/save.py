import torch
import os


MODELS_DIR = os.getenv("MODELS_DIR")


def save_model(model, dir=None, name="model"):
    """
    Saves PyTorch model to 'MODELS_DIR' folder.

    Args:
        model: PyTorch model instance
        model_name (str): base filename (without extension)

    Returns:
        str: full path to saved file
    """

    if dir is not None:
        dir = os.path.join(MODELS_DIR, dir)
    else:
        dir = MODELS_DIR
    
    os.makedirs(dir, exist_ok=True)

    filename = f"{name}.pth"
    save_path = os.path.join(dir, filename)

    torch.save(model, save_path)

    return save_path
