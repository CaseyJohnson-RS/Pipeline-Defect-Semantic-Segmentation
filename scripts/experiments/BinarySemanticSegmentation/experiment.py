import os
import sys

os.environ["MLFLOW_LOGGING_LEVEL"] = "ERROR"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Стандартная библиотека (уже есть в коде)

# Сторонние библиотеки
import mlflow  # noqa: E402
from rich.console import Console  # noqa: E402

# Локальные пакеты/модули (ваш проект)
from scripts.experiments.BinarySemanticSegmentation.config import (  # noqa: E402
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    UNET_MODEL_CONFIG,
    PBS_TRAIN_CONFIG,
    PS_TRAIN_CONFIG,
)
from src.console_input import select_option  # noqa: E402
from src.models import load_unet_model  # noqa: E402
from src.semantic_segmentation import train  # noqa: E402


console = Console()


print("# ============================================================")
print("# Creating Binary Semantic Segmentation UNet Model")
print("# ============================================================\n")


USERNAME = input("Enter your name: ")
os.environ["USER"] = USERNAME

print(f"Hello, {USERNAME}!\n")


with console.status('Connecting to MLFlow...'):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
print("Connected to MLFLow server!")


model = load_unet_model(**UNET_MODEL_CONFIG)
dataset_name = select_option([
    "PS - Pipline Segmentation Augmented",
    "PBS - Pipline Box Segmentation Augmented"
], "Select a dataset: ")

if dataset_name == "PS - Pipline Segmentation Augmented":
    train(model, PS_TRAIN_CONFIG)
else:
    train(model, PBS_TRAIN_CONFIG)