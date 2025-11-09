from dotenv import load_dotenv
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.console_input import confirm
from src.semantic_segmentation import SegmentationDataset, compute_iou, compute_dice, visualize_predictions
from src.tools import save_model, gradient_color, set_seed
from rich.console import Console
from datetime import datetime

load_dotenv()
console = Console()

UNET_MODEL_PREFIX = os.getenv('UNET_MODEL_PREFIX')


def train(model, CONFIG):

    set_seed(CONFIG['seed'])

    with console.status("Loading datasets..."):
        train_ds = SegmentationDataset(
            CONFIG['train_ds']['images'], 
            CONFIG['train_ds']['masks'],
            CONFIG['image_size']
        )

        val_dss = []
        for ds in CONFIG['val_dss']:
            val_dss.append(SegmentationDataset(ds['images'], ds['masks'], CONFIG['image_size']))

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        val_loaders = []
        for val_ds in val_dss:
            val_loaders.append(DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True))
    print("Datasets have been loaded!\n")


    run_name = f"Binary Semantic Segmentation ({CONFIG['dataset']})"

    print(f"Traning {run_name} will start with params: ")
    for key, value in CONFIG['log_params'].items():
        print(f"\t{key:<{30}} {str(value):<{30}}")

    if not confirm("Is everything correct (Y/n)? "):
        exit()
    
    optimizer = CONFIG['optimizer'](
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr']
    )
    criterion = CONFIG['criterion']()

    
    with mlflow.start_run(run_name=run_name):
        print("\nStarting run...")

        # Логируем параметры
        mlflow.log_params(CONFIG['log_params'])

        epochs = CONFIG['epochs']
        val_steps = max(1, len(train_loader) // CONFIG['eval_frequency'])

        for epoch in range(epochs):

            # --- Train ---
            model.train()
            total_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
            for imgs, masks in progress_bar:
                imgs, masks = imgs.to(CONFIG['device']), masks.to(CONFIG['device'])
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # --- Валидация, да посреди эпохи, потому что эпохи долгие ---
                if (progress_bar.n + 1) % val_steps == 0:
                    model.eval()
                    progress_bar.set_postfix({'=': 'Evaluation=='})

                    for val_i, val_loader in enumerate(val_loaders):
                        val_loss, val_iou, val_dice = 0.0, 0.0, 0.0
                        with torch.no_grad():
                            for v_imgs, v_masks in val_loader:
                                v_imgs, v_masks = v_imgs.to(CONFIG['device']), v_masks.to(CONFIG['device'])
                                v_preds = model(v_imgs)

                                v_loss = criterion(v_preds, v_masks)
                                val_loss += v_loss.item()
                                val_iou += compute_iou(v_preds, v_masks)
                                val_dice += compute_dice(v_preds, v_masks)

                        avg_val_loss = val_loss / len(val_loader)
                        avg_val_iou = val_iou / len(val_loader)
                        avg_val_dice = val_dice / len(val_loader)

                        tqdm.write(
                            f"{CONFIG['val_dss'][val_i]['name']}\t"
                            f"Val Loss: {avg_val_loss:.3f}\t"
                            f"IoU: {gradient_color(avg_val_iou, 0, 1)}\t"
                            f"Dice: {gradient_color(avg_val_dice, 0, 1)}"
                        )

                        mlflow.log_metrics({
                            f"{CONFIG['val_dss'][val_i]['name']} val_loss": avg_val_loss,
                            f"{CONFIG['val_dss'][val_i]['name']} val_iou": avg_val_iou,
                            f"{CONFIG['val_dss'][val_i]['name']} val_dice": avg_val_dice
                        }, step=epoch * len(train_loader) + progress_bar.n + 1)
                
                avg_train_loss = total_loss / len(train_loader)
                progress_bar.set_postfix({"avg_train_loss": f"{avg_train_loss:.4f}"})

        print("Training completed!\n")
        
        # --- Save Model ---
        save_model(model, f"{UNET_MODEL_PREFIX}{datetime.now().strftime('%m-%d_%H-%M')}")

        save_model_to_server = confirm("Save the model to the server (Y/n)? ", invalid_response_defaults_to_no=False)
        if save_model_to_server:
            with console.status('Saving model (3-5 mins)...'):
                input_example = np.random.rand(1, 3, 256, 256).astype(np.float32)
                signature = infer_signature(input_example)
                mlflow.pytorch.log_model(model, name="UNetBimarySemanticSegmentation", signature=signature)
            print("Model saved on cloud!")

        # --- Visualize ---
        with console.status("Making visualizations..."):
            vis_path = "predictions.png"
            visualize_predictions(model, val_ds, CONFIG['device'], save_path=vis_path, num_samples=CONFIG['visualization_samples'])
        with console.status("Saving visualizations (5-15 sec)..."):
            mlflow.log_artifact(vis_path)
        print("Visualization saved on cloud!\n")


    