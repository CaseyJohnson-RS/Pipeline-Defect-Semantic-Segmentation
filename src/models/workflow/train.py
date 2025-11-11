import mlflow
from torch import nn
from tqdm import tqdm
from src.console import colored_text

def log_metrics(step: int, metrics: dict):
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics, step=step)


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

    try:
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

                    val_metrics = val_function(model, val_loader, criterion, device, log=True)

                    tqdm.write(
                        "Validation\t" + '\t'.join([f"{k}: {v:.3f}" for k, v in val_metrics.items()])
                    )
                    
                    log_metrics(
                        epoch * len(train_loader) + step,
                        {f"Validation {k}": v for k, v in val_metrics.items()} | {"Train Loss": avg_loss, 'epoch': epoch + step / len(train_loader)}
                    )
    except KeyboardInterrupt:
        print(colored_text("Training interrupted. Returning the model as is.", "yellow"))

    return model