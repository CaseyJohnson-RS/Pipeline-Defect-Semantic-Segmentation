import torch


def _prepare_binary_masks(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Преобразует логиты модели и целевые маски в бинарные карты 0/1.

    Args:
        preds: логиты модели формы (B, 1, H, W)
        targets: целевые маски формы (B, 1, H, W) со значениями 0/1
        threshold: порог для бинаризации предсказаний


    Returns:
        Бинаризованные предсказания и цели (оба тензора типа float)
    """
    # Применяем сигмоиду и бинаризуем предсказания
    binary_preds = (torch.sigmoid(preds) > threshold).float()
    # Целевые маски оставляем как есть (уже 0/1)
    binary_targets = targets.float()

    return binary_preds, binary_targets


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Вычисляет метрику Intersection over Union (IoU) для бинарной сегментации.

    Args:
        preds: логиты модели формы (B, 1, H, W)
        targets: целевые маски формы (B, 1, H, W)
        threshold: порог для бинаризации предсказаний
        eps: малое значение для избежания деления на ноль


    Returns:
        Среднее значение IoU по батчу (float)
    """
    binary_preds, binary_targets = _prepare_binary_masks(preds, targets, threshold)

    intersection = (binary_preds * binary_targets).sum(dim=(1, 2, 3))
    union = (binary_preds + binary_targets - binary_preds * binary_targets).sum(
        dim=(1, 2, 3)
    )

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Вычисляет метрику Dice coefficient для бинарной сегментации.

    Args:
        preds: логиты модели формы (B, 1, H, W)
        targets: целевые маски формы (B, 1, H, W)
        threshold: порог для бинаризации предсказаний
        eps: малое значение для избежания деления на ноль


    Returns:
        Среднее значение Dice по батчу (float)
    """
    binary_preds, binary_targets = _prepare_binary_masks(preds, targets, threshold)

    intersection = (binary_preds * binary_targets).sum(dim=(1, 2, 3))
    total = binary_preds.sum(dim=(1, 2, 3)) + binary_targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (total + eps)
    return dice.mean().item()
