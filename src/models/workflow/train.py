import os
from datetime import datetime
import mlflow
import signal
import torch
from torch import nn
from tqdm import tqdm
from src.console import colored_text, select_option


MODELS_DIR = os.getenv('MODELS_DIR','/')
pause = False
checkpoint_count = 0
train_start_time = None


def signal_handler(signum, frame):
    global pause
    pause = True


def log_metrics(step: int, metrics: dict):
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics, step=step)


def save_model_checkpoint(model):
    """
    Сохраняет модель в директорию, названную по train_start_time,
    с именами checkpoint_1.pth, checkpoint_2.pth и т. д.

    Args:
        model: экземпляр модели PyTorch
    """
    global checkpoint_count, train_start_time

    # Проверяем, что train_start_time задана
    if train_start_time is None:
        raise ValueError("Train didn't start yet!")

    global MODELS_DIR

    # Формируем имя корневой директории на основе train_start_time
    # Формат: YYYYMMDD_HHMMSS (например, 20251112_103000)
    timestamp_str = train_start_time.strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(MODELS_DIR, timestamp_str)

    # Создаём корневую директорию, если её нет
    os.makedirs(root_dir, exist_ok=True)

    # Формируем имя файла: checkpoint_{номер}.pth
    checkpoint_name = f"checkpoint_{checkpoint_count + 1}.pth"
    model_path = os.path.join(root_dir, checkpoint_name)

    # Сохраняем модель (только state_dict — рекомендуемый способ)
    torch.save(model, model_path)

    print(colored_text(f"Checkpoint saved: {model_path}", 'green'))

    # Увеличиваем счётчик
    checkpoint_count += 1


def choose_model_checkpoint(model, device):
    
    global checkpoint_count, train_start_time

    if checkpoint_count == 0:
        return model

    check_point = select_option(
        ["Current model"] + [f"checkpoint_{i + 1}.pth" for i in range(checkpoint_count)],
        "Select model checkpoint as final model: "
    )

    if check_point == "Current model":
        return model

    base_dir = os.getenv("MODELS_DIR")
    timestamp_str = train_start_time.strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(base_dir, timestamp_str)
    return torch.load(os.path.join(root_dir, check_point), device, weights_only=False)


def train(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        device,
        epochs: int,
        val_function: callable,
        val_per_epoch: int = 1,
    ) -> nn.Module:

    model.to(device)
    signal.signal(signal.SIGINT, signal_handler)

    global train_start_time
    train_start_time = datetime.now()

    for epoch in range(epochs):
        model.train()
        cum_loss = 0.0

        tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        step = 0
        val_every_steps = max(1, len(train_loader) // val_per_epoch)

        for batch, targets in tqdm_train_loader:

            step += 1
            
            batch, targets = batch.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            avg_loss = cum_loss / step
            tqdm_train_loader.set_postfix({"loss": f"{avg_loss:.3f}"})

            if step % val_every_steps == 0 and val_loader is not None and val_function is not None and val_per_epoch > 0:
                model.eval()

                val_dict = val_function(model, val_loader, criterion, device, log=True)

                tqdm.write("Validation\t" + val_dict['console_log'])
                
                log_metrics(
                    epoch * len(train_loader) + step,
                    {f"Validation {k}": v for k, v in val_dict['metrics'].items()} | {"Train Loss": avg_loss, 'epoch': epoch + step / len(train_loader)}
                )
            
            global pause
            if pause:
                print(colored_text("Training paused.", "yellow"))
                selected_option = select_option([
                    'Continue',
                    'Save model checkpoint',
                    'Stop training'
                ])

                if selected_option == 'Continue':
                    pause = False
                elif selected_option == 'Save model checkpoint':
                    save_model_checkpoint(model)
                    pause = False
                else:
                    return choose_model_checkpoint(model, device)

                print(colored_text("Training continue...", "yellow"))

    return choose_model_checkpoint(model, device)