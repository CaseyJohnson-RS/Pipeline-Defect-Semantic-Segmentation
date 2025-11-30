from initenv import (
    MLFLOW_TRACKING_URI,
    DATASETS_DIR,
    CONFIG,
)
import os
from mlflow.models import infer_signature
import numpy as np
from rich.console import Console
import shutil
import torch
from src import set_seed
from src.console import confirm, colored_text, print_params_json

from src.models import load_model
from src.losses import get_loss_registry

from src.evaluation.segmentation import evaluate_soft, evaluate_binary 
from src.workflow import train, save_model
from src.data import SegmentationDataset
from src.workflow import tracking

console = Console()
terminal_width = shutil.get_terminal_size().columns


# ====================== Experiment setups ====================================================

print(f"# {'=' * (terminal_width - 4)} #")
print(f"# {CONFIG['experiment_name'] + ': BSS':^{(terminal_width - 4)}} #")
print(f"# {'=' * (terminal_width - 4)} #\n")

# Connect to MLFlow
tracking.connect_server(MLFLOW_TRACKING_URI, CONFIG['experiment_name'])

# Load model (use args from config when creating a new model)
model = load_model(
    CONFIG['model']['name'],
    **CONFIG['model']['args']
)

# Confirm parameters
print("Parameters:")
print_params_json(CONFIG)
if not confirm("Continue? [Y/n] "):
    exit()

# Determine random 
set_seed(CONFIG["seed"])

# Load loss functions registry
supported_criteria = get_loss_registry()

# =============================================================================================


# ===================== Train prerarations ====================================================

# Load dataset

loaders_args = {
    'batch_size': CONFIG["batch_size"],
    'shuffle': True,
    'num_workers': 0,
    'pin_memory': True,
}

train_ds = SegmentationDataset(
    os.path.join(DATASETS_DIR, CONFIG["train_dataset"], "images"),
    os.path.join(DATASETS_DIR, CONFIG["train_dataset"], "masks"),
    CONFIG["image_size"],
)
train_loader = torch.utils.data.DataLoader(train_ds, **loaders_args)

# For validation/evaluation we don't want to shuffle (stable metrics)
val_loaders_args = {**loaders_args, 'shuffle': False}

val_ds = SegmentationDataset(
    os.path.join(DATASETS_DIR, CONFIG["validataion_dataset"], "images"),
    os.path.join(DATASETS_DIR, CONFIG["validataion_dataset"], "masks"),
    CONFIG["image_size"],
)
val_loader = torch.utils.data.DataLoader(val_ds, **val_loaders_args)

if CONFIG["evaluation_dataset"] and len(CONFIG["evaluation_dataset"]) > 0:
    eval_ds = SegmentationDataset(
        os.path.join(DATASETS_DIR, CONFIG["evaluation_dataset"], "images"),
        os.path.join(DATASETS_DIR, CONFIG["evaluation_dataset"], "masks"),
        CONFIG["image_size"],
    )
    eval_loader = torch.utils.data.DataLoader(eval_ds, **val_loaders_args)
else:
    eval_ds = None
    eval_loader = None

print(
    colored_text(
        f"Datasets loaded. Train: {len(train_loader)}, "
        f"Validation: {len(val_loader)}, "
        f"Evaluation {len(eval_loader) if eval_loader else 'None'}"
    )
)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["learning_rate"],
    weight_decay=0.01,
)

if CONFIG["criterion"]["name"] not in supported_criteria:
    raise ValueError(f"Unsupported criterion: {CONFIG['criterion']['name']}")
criterion = supported_criteria[CONFIG["criterion"]["name"]](
    **CONFIG["criterion"]["args"]
)
print(colored_text("Optimizer and criterion initialized"))

# =============================================================================================


# ===================== Train =================================================================

# --- Freeze BatchNorm layers to stabilize training with small batch sizes ---

bn_count = 0
for _m in model.modules():
    if isinstance(_m, torch.nn.BatchNorm2d):
        _m.eval()
        for _p in _m.parameters():
            _p.requires_grad = False
        bn_count += 1
if bn_count:
    print(colored_text(f"Frozen {bn_count} BatchNorm modules"))

run_id = tracking.start_run().info.run_id
tracking.log_params(CONFIG)

trained_model = train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=CONFIG["device"],
    epochs=CONFIG["epochs"],

    train_loader=train_loader,
    val_loader=val_loader,
    eval_loader=eval_loader,
    val_function = evaluate_soft if CONFIG['validation_metrics'] == 'soft' else evaluate_binary,
    eval_function = evaluate_soft if CONFIG['evaluation_metrics'] == 'soft' else evaluate_binary,

    scoring_per_epoch=CONFIG["scoring_per_epoch"],
    save_by_metric=CONFIG["save_by_metric"],
)

tracking.end_run()

print(colored_text("Training completed!\n"))

# =============================================================================================


# ===================== Logging artifacts =====================================================

# Making visualizations
make_viz = confirm(
    "Make visualizations? [Y/n]", invalid_response_defaults_to_no=False
)
if make_viz:
    with console.status("Making visualizations..."):
        vis_path = "predictions.png"
        from src.workflow.visualizations import visualize_predictions

        visualize_predictions(
            trained_model,
            eval_ds if eval_ds else val_ds,
            CONFIG["device"],
            save_path=vis_path,
            num_samples=CONFIG["visualization_samples"],
            threshold=0.45,
        )
    tracking.log_artifact(vis_path, run_id=run_id)

# Send model to server
save_model_to_server = CONFIG['track_experiment'] and confirm(
    "Save the model to the server? [Y/n]", invalid_response_defaults_to_no=False
)
if save_model_to_server:
    input_example = np.random.rand(1, 3, 256, 256).astype(np.float32)
    signature = infer_signature(input_example)
    tracking.log_model(model, name=CONFIG['model']['name'], signature=signature, run_id=run_id)

# Save model to local
save_model_local = confirm(
    "Save the model locally? [Y/n]", invalid_response_defaults_to_no=False
)
if save_model_local:
    
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_model(
        trained_model,
        CONFIG['model']['name'],
        f"{CONFIG['train_dataset']}_{ts}"
    )

    print(colored_text("Model saved locally" ))

# =============================================================================================
