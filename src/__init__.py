from dotenv import load_dotenv
import numpy as np
import random
from pathlib import Path
import torch
import yaml
from typing import Optional

load_dotenv()


def set_seed(SEED):
    """Sets seed for reproducible results."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def check_cuda_available():
    """Prints information about CUDA and GPU availability."""
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        return True
    return False


def load_yaml(filepath: str | Path) -> Optional[dict]:
    """
    Loads a YAML file and returns its contents as a dictionary.

    Args:
        filepath: path to YAML file (string or Path object)

    Returns:
        Dictionary with YAML file data or None on error

    Examples:
        >>> data = load_yaml_file("config.yaml")
        >>> if data is not None:
        ...     print(data)
    """
    file_path = Path(filepath)

    # Check if file exists
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    # Check that it's a file (not a directory)
    if not file_path.is_file():
        print(f"The path is not a file: {file_path}")
        return None

    try:
        with file_path.open('r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            
        # Check that loaded data is a dictionary
        if not isinstance(data, dict):
            print(f"File content is not a dictionary: {file_path}")
            return None
            
        return data

    except yaml.YAMLError as e:
        print(f"YAML parsing error in file {file_path}: {e}")
        return None
    except PermissionError:
        print(f"No read permission for file: {file_path}")
        return None
    except UnicodeDecodeError as e:
        print(f"UTF-8 decoding error in file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading file {file_path}: {e}")
        return None

