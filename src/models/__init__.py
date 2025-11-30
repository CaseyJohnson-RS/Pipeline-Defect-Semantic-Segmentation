import os
from pathlib import Path
from segmentation_models_pytorch import Unet
import torch
from .UNetAttention import UNetAttention
from src.console.input import confirm, select_option
from src.console import colored_text


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

    models = [str(model_path).split('\\')[-1] for model_path in _get_available_model_paths(model_type)]
    if not models or len(models) == 0:
        raise FileNotFoundError(f"Models of type {model_type} do not exist")

    model_name = select_option(models, "Select a file: ")
    model_path = MODELS_DIR / model_type / model_name
    return torch.load(model_path, weights_only=False)


def load_model(model_type: str | None = None, **model_kwargs):
    """
    Load or create a model.

    model_kwargs may contain an optional boolean key `freeze_encoder`.
    If present and True, the function will freeze `model.encoder` parameters
    (useful for transfer learning).
    """
    # Pop freeze flag so it isn't passed to model constructor
    freeze_encoder = bool(model_kwargs.pop("freeze_encoder", False))

    if model_type is None or confirm("Load existing model? [Y/n]"):
        model = _load_model_from_file()
    else:
        model = SUPPORTED_MODELS[model_type](**model_kwargs)

    # Apply encoder freeze if requested
    if freeze_encoder:
        try:
            for p in model.encoder.parameters():
                p.requires_grad = False
            print(colored_text("Encoder frozen for transfer learning", "cyan"))
        except Exception:
            print(colored_text("Warning: failed to freeze encoder (no attribute)", "red"))
    else:
        print(colored_text("Encoder UNFROZEN (full fine-tuning)", "yellow"))

    return model
