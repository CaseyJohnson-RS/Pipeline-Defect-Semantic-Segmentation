import torch
def _prepare_tensors(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Приводит логиты и маски к бинарным картам 0/1 для метрик.
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()  # уже 0/1
    return preds, targets

def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds, targets = _prepare_tensors(preds, targets, threshold)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def compute_dice(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds, targets = _prepare_tensors(preds, targets, threshold)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / 
            (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6)).mean().item()
