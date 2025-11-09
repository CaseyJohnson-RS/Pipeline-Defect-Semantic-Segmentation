import torch


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Вычисляет средний IoU для батча.
    """
    preds = torch.sigmoid(preds) if preds.min() < 0 or preds.max() > 1 else preds
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def compute_dice(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Вычисляет средний Dice (F1) для батча.
    """
    preds = torch.sigmoid(preds) if preds.min() < 0 or preds.max() > 1 else preds
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6)
    return dice.mean().item()