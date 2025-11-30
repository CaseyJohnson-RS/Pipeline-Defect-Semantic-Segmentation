"""
High-level training loop with MLflow logging and interactive pause / resume.
"""
from __future__ import annotations

import os
import signal
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.console import colored_text, select_option

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
MetricDict = Dict[str, float]
ValFunc = Callable[
    [nn.Module, DataLoader, nn.Module, torch.device, bool, str, str],
    Dict[str, Any],
]

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


# --------------------------------------------------------------------------- #
# State & checkpointing
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TrainState:
    """Mutable training state that survives between epochs / pauses."""

    start_time: datetime
    total_steps: int = 0
    best_metric_value: float = 0.0
    checkpoint_counter: int = 0


class CheckpointManager:
    """Handles save / load logic for full model checkpoints."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    def save(self, model: nn.Module, state: TrainState) -> Path:
        """Save model and return path."""
        state.checkpoint_counter += 1
        fname = self.root_dir / f"checkpoint_{state.checkpoint_counter}.pth"
        torch.save(model, fname)  # whole model (recommended way)
        return fname

    # --------------------------------------------------------------------- #
    def list_checkpoints(self) -> list[Path]:
        """Return sorted list of existing checkpoint files."""
        return sorted(self.root_dir.glob("checkpoint_*.pth"))

    # --------------------------------------------------------------------- #
    def load(self, path: Path, device: torch.device) -> nn.Module:
        """Load full model onto device."""
        return torch.load(path, map_location=device, weights_only=False)


# --------------------------------------------------------------------------- #
# Metrics & MLflow helpers
# --------------------------------------------------------------------------- #
def _log_metrics(step: int, metrics: MetricDict, prefix: str = "") -> None:
    """Log dict to MLflow (if run is active)."""
    if mlflow.active_run() is None:
        return
    if prefix:
        metrics = {f"{prefix} {k}": v for k, v in metrics.items()}
    mlflow.log_metrics(metrics, step=step)


def _evaluate(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    val_loader: Optional[DataLoader],
    eval_loader: Optional[DataLoader],
    val_fn: Optional[ValFunc],
    eval_fn: Optional[ValFunc],
    step: int,
) -> MetricDict:
    """Run validation and/or evaluation and return unified metrics dict."""
    metrics: MetricDict = {}

    if val_loader is not None and val_fn is not None:
        d = val_fn(
            model,
            val_loader,
            criterion,
            device,
            log=True,
            prefix="Validation",
            colour="#8600bf",
        )
        tqdm.write("Validation\t" + d["console_log"])
        metrics.update(d["metrics"])
        _log_metrics(step, d["metrics"], "Validation")

    if eval_loader is not None and eval_fn is not None:
        d = eval_fn(
            model,
            eval_loader,
            criterion,
            device,
            log=True,
            prefix="Evaluation",
            colour="#0099bf",
        )
        tqdm.write("Evaluation\t" + d["console_log"])
        metrics.update(d["metrics"])
        _log_metrics(step, d["metrics"], "Evaluation")

    return metrics


# --------------------------------------------------------------------------- #
# Interactive pause
# --------------------------------------------------------------------------- #
_pause_event = threading.Event()


def _install_sigint_handler() -> None:
    """Convert SIGINT into a threading.Event."""

    def _handler(_sig, _frame) -> None:
        _pause_event.set()

    signal.signal(signal.SIGINT, _handler)


def _wait_user_choice(model: nn.Module, ckpt_mgr: CheckpointManager, state: TrainState) -> bool:
    """
    Return True  -> continue training
           False -> stop training (return from train())
    """
    print(colored_text("\nTraining paused.", "yellow"))
    choice = select_option(
        ["Continue", "Save model checkpoint", "Stop training"]
    )

    if choice == "Continue":
        _pause_event.clear()
        print(colored_text("Training continue...", "yellow"))
        return True

    if choice == "Save model checkpoint":
        ckpt_mgr.save(model, state)
        _pause_event.clear()
        return True

    # Stop
    return False


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    eval_loader: Optional[DataLoader] = None,
    val_function: Optional[ValFunc] = None,
    eval_function: Optional[ValFunc] = None,
    scoring_per_epoch: int = 1,
    save_by_metric: Optional[str] = None,
) -> nn.Module:
    """
    Train loop with:
        * MLflow logging
        * periodic validation / evaluation
        * interactive pause (Ctrl-C) → continue / save / stop
        * automatic best-model checkpointing
        * final checkpoint selection

    Returns
    -------
    nn.Module
        The model chosen by user (current or one of checkpoints).
    """
    _install_sigint_handler()
    model.to(device)

    # ----------------------------------------------------------- #
    state = TrainState(start_time=datetime.now())
    ckpt_dir = DEFAULT_MODELS_DIR / state.start_time.strftime("%Y%m%d_%H%M%S")
    ckpt_mgr = CheckpointManager(ckpt_dir)
    shall_continue = True

    # ----------------------------------------------------------- #
    for epoch in range(1, epochs + 1):
        model.train()
        cum_loss = 0.0
        step_in_epoch = 0
        val_every = max(1, len(train_loader) // scoring_per_epoch)

        pbar = tqdm(
            train_loader,
            desc=f"Train [{epoch}/{epochs}]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch, targets in pbar:
            step_in_epoch += 1
            state.total_steps += 1

            batch, targets = batch.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            avg_loss = cum_loss / step_in_epoch
            pbar.set_postfix(loss=f"{avg_loss:.3f}")

            # -------------- validation / checkpoint -------------- #
            if step_in_epoch % val_every == 0 and scoring_per_epoch > 0:
                model.eval()
                metrics = _evaluate(
                    model,
                    criterion,
                    device,
                    val_loader,
                    eval_loader,
                    val_function,
                    eval_function,
                    state.total_steps,
                )

                if save_by_metric and save_by_metric in metrics:
                    if metrics[save_by_metric] > state.best_metric_value:
                        state.best_metric_value = metrics[save_by_metric]
                        fname = ckpt_mgr.save(model, state)
                        tqdm.write(f"Checkpoint saved (new best): {fname}")

                _log_metrics(
                    state.total_steps,
                    {"Train Loss": avg_loss, "epoch": epoch + step_in_epoch / len(train_loader)},
                )

            # --------------------- pause ? --------------------- #
            if _pause_event.is_set():
                if not shall_continue:
                    break
                shall_continue = _wait_user_choice(model, ckpt_mgr, state)
                

        # ------------------- epoch end ------------------- #
        if _pause_event.is_set():
            if not shall_continue:
                break
            shall_continue = _wait_user_choice(model, ckpt_mgr, state)
            

    # ----------------------------------------------------------- #
    # Final model selection
    # ----------------------------------------------------------- #
    options: list[str] = ["Current model"] + [
        f"checkpoint_{i + 1}.pth" for i in range(state.checkpoint_counter)
    ]
    chosen = select_option(options, "Select model checkpoint as final model: ")

    if chosen != "Current model":
        model = ckpt_mgr.load(ckpt_dir / chosen, device)

    # final evaluation
    _evaluate(
        model,
        criterion,
        device,
        val_loader,
        eval_loader,
        val_function,
        eval_function,
        state.total_steps + 1,
    )
    return model