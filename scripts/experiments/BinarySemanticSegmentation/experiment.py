# ========================== Настройки пути и окружения ========================= #

from dotenv import load_dotenv
import os
import sys


os.environ["MLFLOW_LOGGING_LEVEL"] = "ERROR"

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.relpath(current_dir, start=os.getcwd())

PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

# =================================================================================#

import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
import shutil  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from src import load_yaml_file, set_seed  # noqa: E402
from src.console import confirm, input_with_default, colored_text  # noqa: E402
from src.models.factories import load_unet  # noqa: E402
from src.models.losses import WeightedCrossEntropyLoss  # noqa: E402
from src.models.evaluation import semantic_segmentation_evaluation  # noqa: E402
from src.models.workflow import train, save_model  # noqa: E402
from src.data import SegmentationDataset  # noqa: E402


EXPCFG = load_yaml_file(os.path.join(relative_path, "config.yaml"))
EXPCFG["device"] = "cuda" if torch.cuda.is_available() else "cpu"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
DATASETS_DIR = os.getenv("DATASETS_DIR")
UNET_MODEL_PREFIX = os.getenv("UNET_MODEL_PREFIX", "unet_bss_")

supported_criteria = {
    "WeightedCrossEntropyLoss": WeightedCrossEntropyLoss,
    "BCELoss": torch.nn.BCELoss,
}

console = Console()
terminal_width = shutil.get_terminal_size().columns

# ---------------------------------------------------------------------------------#

print(f"# {'=' * (terminal_width - 4)} #")
print(f"# {EXPERIMENT_NAME + ': BSS':^{(terminal_width - 4)}} #")
print(f"# {'=' * (terminal_width - 4)} #\n")

# Приветствие пользователя
USERNAME = os.getenv("USERNAME")
if not USERNAME:
    USERNAME = input("Enter your name: ")
    os.environ["USERNAME"] = USERNAME
print(colored_text(f"Hello, {USERNAME}!", "green"))

# Подключение к MLFlow
with console.status("Connecting to MLFlow..."):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
print(colored_text("Connected to MLFLow server!", "green"))

# Загрузка модели
model = load_unet(**EXPCFG["model"]["args"]).to(EXPCFG["device"])

# Задаём имя запуска
run_name = input_with_default(
    prompt="Enter run name", default="Binary Semantic Segmentation"
)
print(colored_text(f"Run name set to: {run_name}", "green"))

# Подтверждение параметров
print("\nTraining will start with the following parameters:")
print(
    colored_text(yaml.dump(EXPCFG, default_flow_style=False, sort_keys=False), "orange")
)
if not confirm("Is everything correct (default: No)? "):
    exit()

# Загрузка датасета
print(colored_text("\nConfirmed", "green"))

set_seed(EXPCFG["seed"])

with console.status("Loading dataset..."):
    train_ds = SegmentationDataset(
        os.path.join(DATASETS_DIR, EXPCFG["dataset"], "images", "train"),
        os.path.join(DATASETS_DIR, EXPCFG["dataset"], "masks", "train"),
        EXPCFG["image_size"],
    )
    val_ds = SegmentationDataset(
        os.path.join(DATASETS_DIR, EXPCFG["dataset"], "images", "val"),
        os.path.join(DATASETS_DIR, EXPCFG["dataset"], "masks", "val"),
        EXPCFG["image_size"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=EXPCFG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=EXPCFG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
print(colored_text(f"Dataset loaded. Train: {len(train_loader)}, Validation: {len(val_loader)}", "green"))

# Инициализация оптимизатора и функции потерь
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=EXPCFG["learning_rate"]
)
if EXPCFG["criterion"]["name"] not in supported_criteria:
    raise ValueError(f"Unsupported criterion: {EXPCFG['criterion']['name']}")
criterion = supported_criteria[EXPCFG["criterion"]["name"]](
    **EXPCFG["criterion"]["args"]
)
print(colored_text("Optimizer and criterion initialized", "green"))


# Запуск обучения
with mlflow.start_run(run_name=run_name):
    mlflow.log_params(EXPCFG)

    trained_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=EXPCFG["device"],
        epochs=EXPCFG["epochs"],
        val_function=semantic_segmentation_evaluation,
        val_per_epoch=EXPCFG["val_per_epoch"],
    )

    print(colored_text("Training completed!\n", "green"))

    # Визуализируем предсказания
    with console.status("Making visualizations..."):
        vis_path = "predictions.png"
        from src.models.workflow.visualizations import visualize_predictions

        visualize_predictions(
            trained_model,
            val_ds,
            EXPCFG["device"],
            save_path=vis_path,
            num_samples=EXPCFG["visualization_samples"],
        )
    with console.status("Saving visualizations (5-15 sec)..."):
        mlflow.log_artifact(vis_path)
    print(colored_text("Visualization saved on cloud!\n", "green"))

    # Отправка модели на сервер и/или сохранение локально
    save_model_to_server = confirm("Save the model to the server (Y/n)? ", invalid_response_defaults_to_no=False)
    if save_model_to_server:
        with console.status('Saving model (3-5 mins)...'):
            input_example = np.random.rand(1, 3, 256, 256).astype(np.float32)
            signature = infer_signature(input_example)
            mlflow.pytorch.log_model(trained_model, name="UNetBimarySemanticSegmentation", signature=signature)
        print(colored_text("Model saved on cloud!", "green"))
    
    save_model_local = confirm("Save the model locally (default: Yes)? ", default_yes=True)
    if save_model_local:
        save_model(trained_model, f"{UNET_MODEL_PREFIX}{EXPCFG['dataset']}_{run_name.replace(' ', '_')}")
        print(colored_text("Model saved locally!", "green"))
    