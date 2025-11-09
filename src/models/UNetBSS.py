from pathlib import Path
import torch
from segmentation_models_pytorch import Unet
from src.console_input import confirm, select_option
from dotenv import load_dotenv
import os


load_dotenv()

UNET_MODEL_PREFIX = os.getenv('UNET_MODEL_PREFIX')
MODELS_DIRECTORY = Path(os.getenv('MODELS_DIRECTORY'))

torch.serialization.add_safe_globals([Unet])


def get_available_model_paths() -> list[Path]:
    """
    Returns list of model file paths in MODELS_DIRECTORY that start with MODEL_PREFIX.
    
    Returns:
        List of Path objects for available model files.
    """
    if not MODELS_DIRECTORY.exists():
        return []
    
    return [
        file_path for file_path in MODELS_DIRECTORY.iterdir()
        if file_path.is_file() and file_path.name.startswith(UNET_MODEL_PREFIX)
    ]



def load_unet_model(
    encoder_name: str,
    in_channels: int,
    classes: int,
    device: torch.device,
    default_encoder_weights: str | None = None,
    **model_kwargs
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
    should_load_existing = confirm("Load existing model (Y/n, default=No)? ")
    
    if should_load_existing and available_models:
        # Let user select from available models
        model_names = [model_path.name for model_path in available_models]
        selected_name = select_option(model_names, "Select a model: ")
        
        if selected_name is None:
            raise ValueError("No model selected")
            
        model_path = MODELS_DIRECTORY / selected_name
        print(f"Loading '{model_path}'... ", end="")
        
        # Initialize model without pretrained weights
        model = Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            **model_kwargs
        )
        model.to(device)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        try:
            model.load_state_dict(state_dict)
        except TypeError:
            model = state_dict
        
        print("success!")
        
    else:
        # Use default pretrained weights
        weights_desc = default_encoder_weights or "None"
        print(f"Loading weights '{weights_desc}'... ", end="")
        
        model = Unet(
            encoder_name=encoder_name,
            encoder_weights=default_encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **model_kwargs
        )
        model.to(device)
        print("success!")
    
    # Freeze encoder for transfer learning
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    return model
