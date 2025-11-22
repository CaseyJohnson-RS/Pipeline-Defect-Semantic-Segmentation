import torch
from ._base import probs_from_logits, normalize_soft_target, _EPS
from typing import Tuple

def _prepare(preds: torch.Tensor, targets: torch.Tensor, eps: float = _EPS
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = probs_from_logits(preds, eps)
    targets = normalize_soft_target(targets)
    return preds, targets

# ----------------------------------------------------------
def iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = _EPS) -> float:
    p, t = _prepare(preds, targets, eps)
    inter = torch.minimum(p, t).sum(dim=(1, 2, 3))
    union = torch.maximum(p, t).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()

def dice(preds: torch.Tensor, targets: torch.Tensor, eps: float = _EPS) -> float:
    p, t = _prepare(preds, targets, eps)
    inter = (p * t).sum(dim=(1, 2, 3))
    total = (p ** 2 + t ** 2).sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (total + eps)).mean().item()

def tpr(preds: torch.Tensor, targets: torch.Tensor, eps: float = _EPS) -> float:
    p, t = _prepare(preds, targets, eps)
    tp = (p * t).sum(dim=(1, 2, 3))
    denom = t.sum(dim=(1, 2, 3))
    return ((tp + eps) / (denom + eps)).mean().item()

def tnr(preds: torch.Tensor, targets: torch.Tensor, eps: float = _EPS) -> float:
    p, t = _prepare(preds, targets, eps)
    tn = ((1 - p) * (1 - t)).sum(dim=(1, 2, 3))
    denom = (1 - t).sum(dim=(1, 2, 3))
    return ((tn + eps) / (denom + eps)).mean().item()