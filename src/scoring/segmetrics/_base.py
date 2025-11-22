import torch

_EPS = 1e-6

def probs_from_logits(preds: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Convert logits to probabilities if needed."""
    if preds.min() < 0.0 or preds.max() > 1.0 + eps:
        preds = torch.sigmoid(preds)
    return preds

def normalize_soft_target(targets: torch.Tensor) -> torch.Tensor:
    """Bring targets to [0,1] range."""
    targets = targets.float()
    if targets.max() > 1.0:
        targets = targets / 255.0
    return torch.clamp(targets, 0.0, 1.0)