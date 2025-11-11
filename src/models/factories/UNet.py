from pathlib import Path
import torch
from segmentation_models_pytorch import Unet
from src.console import confirm, select_option, colored_text
import os


UNET_MODEL_PREFIX = os.getenv("UNET_MODEL_PREFIX")
MODELS_DIR = Path(os.getenv("MODELS_DIR"))

# torch.serialization.add_safe_globals([Unet])


def get_available_model_paths() -> list[Path]:
    """
    Returns list of model file paths in MODELS_DIR that start with MODEL_PREFIX.

    Returns:
        List of Path objects for available model files.
    """
    if not MODELS_DIR.exists():
        return []

    return [
        file_path
        for file_path in MODELS_DIR.iterdir()
        if file_path.is_file() and file_path.name.startswith(UNET_MODEL_PREFIX)
    ]


def load_unet(
    encoder_name: str,
    in_channels: int,
    classes: int,
    encoder_weights: str | None = None,
    **model_kwargs,
) -> Unet:
    """
    Loads a Unet model either from a saved checkpoint or with pretrained weights.

    Args:
        encoder_name: Name of the encoder architecture.
        in_channels: Number of input channels.
        classes: Number of output classes.
        device: Device to move the model to (e.g., 'cpu' or 'cuda').
        default_encoder_weights: Pretrained weights to use if not loading existing model.
        **model_kwargs: Additional keyword arguments passed to Unet constructor.

    Returns:
        Loaded and configured Unet model.
    Raises:
        FileNotFoundError: If selected model file doesn't exist.
        RuntimeError: If model loading fails.
    """
    # Check available models
    available_models = get_available_model_paths()

    # Ask user whether to load existing model
    should_load_existing = confirm("Load existing model (default = No)? ")

    if should_load_existing and available_models:
        # Let user select from available models
        model_names = [model_path.name for model_path in available_models]
        selected_name = select_option(model_names, "Select a model: ")

        if selected_name is None:
            raise ValueError("No model selected")

        model_path = MODELS_DIR / selected_name
        print(f"Loading '{model_path}'... ", end="")

        # Initialize model without pretrained weights
        model = Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            **model_kwargs,
        )

        # Load state dict
        state_dict = torch.load(model_path, weights_only=False)

        try:
            model.load_state_dict(state_dict)
        except TypeError:
            model = state_dict

        print(colored_text(f"Loaded '{model_path}'", "green"))

    else:
        # Use default pretrained weights
        weights_desc = encoder_weights or "None"

        model = Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **model_kwargs,
        )

        print(colored_text(f"Loaded '{weights_desc}'", "green"))

    # Freeze encoder for transfer learning
    for param in model.encoder.parameters():
        param.requires_grad = False

    return model
