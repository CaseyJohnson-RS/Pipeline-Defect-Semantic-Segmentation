#=========== Setup root dir =====================================================

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.relpath(current_dir, start=os.getcwd())

PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#================================================================================


#=============== Setup environment variables ====================================

from dotenv import load_dotenv # noqa: E402
from src.console import colored_text # noqa: E402
load_dotenv()

os.environ["MLFLOW_LOGGING_LEVEL"] = "ERROR"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATASETS_DIR = os.getenv("DATASETS_DIR",'datasets')
USERNAME = os.getenv("USERNAME")

if not USERNAME:
    USERNAME = input("Enter your name: ")
    os.environ["USERNAME"] = USERNAME
    print(colored_text(f"Hello, {USERNAME}!"))

#================================================================================


#=============== Load config ====================================================

from pathlib import Path # noqa: E402
from src import load_yaml # noqa: E402
import torch # noqa: E402

config_path = os.path.join(relative_path, "config.yaml")

if Path(config_path).exists():
    CONFIG = load_yaml(config_path)
    if 'device' not in CONFIG or CONFIG['device'] == 'auto':
        CONFIG['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    if 'track_experiment' in CONFIG and CONFIG['track_experiment']:
        os.environ["TRACK_EXPERIMENT"] = 'True'
else:
    CONFIG = {}
    colored_text(f"Missing config file ({config_path})!", "red")
    raise FileNotFoundError(f"Missing config file ({config_path})!")

#================================================================================

#=============== Export variables ===============================================

__all__ = [
    "PROJECT_ROOT",
    "MLFLOW_TRACKING_URI",
    "DATASETS_DIR",
    "USERNAME",
    "CONFIG",
]

#================================================================================

