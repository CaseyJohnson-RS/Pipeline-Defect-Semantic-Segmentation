import torch
from ._base import probs_from_logits, _EPS
from typing import Tuple

def _prepare(preds: torch.Tensor, targets: torch.Tensor,
             threshold: float = 0.5, eps: float = _EPS
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = probs_from_logits(preds, eps)
    preds = (preds > threshold).float()
    targets = (targets > 0.5).float()
    return preds, targets

# ----------------------------------------------------------
def iou(preds: torch.Tensor, targets: torch.Tensor,
        threshold: float = 0.5, eps: float = _EPS) -> float:
    p, t = _prepare(preds, targets, threshold, eps)
    inter = (p * t).sum(dim=(1, 2, 3))
    union = (p + t - p * t).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()

def dice(preds: torch.Tensor, targets: torch.Tensor,
         threshold: float = 0.5, eps: float = _EPS) -> float:
    p, t = _prepare(preds, targets, threshold, eps)
    inter = (p * t).sum(dim=(1, 2, 3))
    total = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (total + eps)).mean().item()

def tpr(preds: torch.Tensor, targets: torch.Tensor,
        threshold: float = 0.5, eps: float = _EPS) -> float:
    p, t = _prepare(preds, targets, threshold, eps)
    tp = (p * t).sum(dim=(1, 2, 3))
    fn = ((1 - p) * t).sum(dim=(1, 2, 3))
    return ((tp + eps) / (tp + fn + eps)).mean().item()

def tnr(preds: torch.Tensor, targets: torch.Tensor,
        threshold: float = 0.5, eps: float = _EPS) -> float:
    p, t = _prepare(preds, targets, threshold, eps)
    tn = ((1 - p) * (1 - t)).sum(dim=(1, 2, 3))
    fp = (p * (1 - t)).sum(dim=(1, 2, 3))
    return ((tn + eps) / (tn + fp + eps)).mean().item()