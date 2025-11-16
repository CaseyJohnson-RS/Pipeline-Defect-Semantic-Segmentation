from pathlib import Path
import torch
from src.models.unet_attention import UNetAttention
from src.console import confirm, select_option, colored_text
import os


UNET_ATTENTION_MODEL_PREFIX = os.getenv("UNET_ATTENTION_MODEL_PREFIX", "unet_attn_")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


def get_available_model_paths(prefix: str = UNET_ATTENTION_MODEL_PREFIX) -> list[Path]:
    """
    Returns list of model file paths in MODELS_DIR that start with prefix.
    """
    if not MODELS_DIR.exists():
        return []

    return [
        file_path
        for file_path in MODELS_DIR.iterdir()
        if file_path.is_file() and file_path.name.startswith(prefix)
    ]


def load_unet_attention(
    encoder_name: str = "resnet34",
    in_channels: int = 3,
    classes: int = 1,
    encoder_weights: str | None = "imagenet",
    dropout_rate: float = 0.3,
    freeze_encoder: bool = True,
    **model_kwargs,
) -> UNetAttention:
    """
    Loads a UNetAttention model either from a saved checkpoint or with pretrained weights.

    Args:
        encoder_name: Name of the encoder architecture (e.g., 'resnet34', 'efficientnet-b3').
        in_channels: Number of input channels.
        classes: Number of output classes.
        encoder_weights: Pretrained weights for encoder ('imagenet' or None).
        dropout_rate: Dropout probability in bottleneck (0.0-1.0).
        freeze_encoder: Whether to freeze encoder weights for transfer learning.
        **model_kwargs: Additional arguments passed to UNetAttention constructor.

    Returns:
        Loaded and configured UNetAttention model.
    """
    # Check available models
    available_models = get_available_model_paths()

    # Ask user whether to load existing model
    should_load_existing = confirm("Load existing UNetAttention model (default: No)? ")

    if should_load_existing and available_models:
        if len(available_models) == 0:
            print(colored_text("No available models found.", "red"))
            raise FileNotFoundError("No available models to load.")

        # Let user select from available models
        model_names = [model_path.name for model_path in available_models]
        selected_name = select_option(model_names, "Select a model: ")

        if selected_name is None:
            raise ValueError("No model selected")

        model_path = MODELS_DIR / selected_name
        print(f"Loading '{model_path}'... ", end="")

        # Initialize model WITHOUT pretrained weights (will load from checkpoint)
        model = UNetAttention(
            encoder_name=encoder_name,
            encoder_weights=None,  # Важно: None при загрузке checkpoint
            in_channels=in_channels,
            classes=classes,
            dropout_rate=dropout_rate,
            **model_kwargs,
        )

        # Load entire model or state dict
        checkpoint = torch.load(model_path, weights_only=False)
        
        try:
            model.load_state_dict(checkpoint)
        except TypeError:
            model = checkpoint

        print(colored_text(f"Loaded '{model_path}'", "green"))

    else:
        # Create NEW model with pretrained encoder
        weights_desc = encoder_weights or "random initialization"

        model = UNetAttention(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            dropout_rate=dropout_rate,
            **model_kwargs,
        )

        print(colored_text(f"Initialized with encoder weights '{weights_desc}'", "green"))

    # Freeze encoder for transfer learning (по умолчанию)
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print(colored_text("Encoder frozen for transfer learning", "cyan"))
    else:
        print(colored_text("Encoder UNFROZEN (full fine-tuning)", "yellow"))

    return model