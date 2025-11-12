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


def compute_tpr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Вычисляет True Positive Rate (Recall) для бинарной сегментации.
    Доля корректно угаданных пикселей внутри целевой маски.

    Args:
        preds: логиты модели формы (B, 1, H, W)
        targets: целевые маски формы (B, 1, H, W)
        threshold: порог для бинаризации предсказаний
        eps: малое значение для избежания деления на ноль


    Returns:
        Среднее значение TPR по батчу (float)
    """
    binary_preds, binary_targets = _prepare_binary_masks(preds, targets, threshold)

    tp = (binary_preds * binary_targets).sum(dim=(1, 2, 3))  # True Positives
    fn = ((1 - binary_preds) * binary_targets).sum(dim=(1, 2, 3))  # False Negatives

    tpr = (tp + eps) / (tp + fn + eps)  # TPR = TP / (TP + FN)
    return tpr.mean().item()


def compute_tnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Вычисляет True Negative Rate (Specificity) для бинарной сегментации.
    Доля корректно угаданных пикселей вне целевой маски.


    Args:
        preds: логиты модели формы (B, 1, H, W)
        targets: целевые маски формы (B, 1, H, W)
        threshold: порог для бинаризации предсказаний
        eps: малое значение для избежания деления на ноль


    Returns:
        Среднее значение TNR по батчу (float)
    """
    binary_preds, binary_targets = _prepare_binary_masks(preds, targets, threshold)


    tn = ((1 - binary_preds) * (1 - binary_targets)).sum(dim=(1, 2, 3))  # True Negatives
    fp = (binary_preds * (1 - binary_targets)).sum(dim=(1, 2, 3))  # False Positives

    tnr = (tn + eps) / (tn + fp + eps)  # TNR = TN / (TN + FP)
    return tnr.mean().item()
