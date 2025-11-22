import os
from pathlib import Path
from segmentation_models_pytorch import Unet
import torch
from .UNetAttention import UNetAttention
from src.console.input import confirm, select_option


MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
SUPPORTED_MODELS = {"UNetAttention": UNetAttention, "UNet": Unet}


def _get_available_model_paths(model_type: str) -> list[Path]:
    """
    Returns list of model file paths in MODELS_DIR by models' type.
    """

    dir = Path(os.path.join(MODELS_DIR, model_type))

    if not dir.exists():
        return []

    return [file_path for file_path in dir.iterdir() if file_path.is_file()]


def _load_model_from_file():
    model_type = select_option(list(SUPPORTED_MODELS.keys()), "Select a model type: ")

    models = _get_available_model_paths(model_type)
    if not models or len(models) == 0:
        raise FileNotFoundError(f"Models of type {model_type} do not exist")

    model_name = select_option(models, "Select a file: ")
    model_path = MODELS_DIR / model_type / model_name
    return torch.load(model_path, weights_only=False)


def load_model(model_type: str | None, **model_kwargs):
    if model_type is None or confirm("Load existing model? [Y/n]"):
        return _load_model_from_file()

    model = SUPPORTED_MODELS[model_type](**model_kwargs)
    return model
