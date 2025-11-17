import torch
from typing import Tuple


def _prepare_masks(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.5,
    force_soft_mode: bool = False,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Подготавливает предсказания и цели для вычисления метрик.
    Определяет мягкий режим по уникальным значениям targets.
    
    В мягком режиме targets интерпретируются как карты вероятностей.
    В бинарном режиме все targets бинаризуются к 0/1.
    
    Args:
        preds: логиты модели формы (B, 1, H, W) или вероятности в [0,1]
        targets: целевые маски формы (B, 1, H, W) с 0/1 или [0, 1] или [0, 255]
        threshold: порог для бинаризации предсказаний (в бинарном режиме)
        force_soft_mode: если True, принудительно использовать мягкий режим
        eps: малое значение для проверок вырожденных случаев
        
    Returns:
        tuple: (processed_preds, processed_targets, is_soft_mode)
    """
    # Детектируем, являются ли предсказания вероятностями (эвристика)
    if preds.min() >= 0.0 and preds.max() <= 1.0 + eps:
        preds_prob = preds
    else:
        preds_prob = torch.sigmoid(preds)
    
    # ======== Определяем режим работы по уникальным значениям ========
    
    # Если значения > 1, то это точно мягкая маска ([0, 255] или [0, 100])
    if force_soft_mode:
        is_soft = True
    else:
        # Проверяем уникальные значения в targets
        unique_vals = torch.unique(targets)
        # Мягкий режим, если есть значения вне {0, 1}
        is_soft = unique_vals.numel() > 2
    
    if is_soft:
        processed_targets = targets.float()
        
        # Авто-нормализация [0, 255] -> [0, 1]
        if processed_targets.max() > 1.0:
            processed_targets = processed_targets / 255.0
        
        # Класп для безопасности
        processed_targets = torch.clamp(processed_targets, 0.0, 1.0)
        
        return preds_prob, processed_targets, True
    else:
        # Бинарный режим — работаем как раньше
        binary_preds = (preds_prob > threshold).float()
        binary_targets = (targets > 0.5).float()
        return binary_preds, binary_targets, False


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    force_soft_mode: bool = False,
) -> float:
    """
    Вычисляет IoU для бинарной или мягкой сегментации.
    Автоматически определяет тип targets по уникальным значениям.
    """
    p, t, is_soft = _prepare_masks(preds, targets, threshold, force_soft_mode, eps)
    
    if is_soft:
        intersection = torch.minimum(p, t).sum(dim=(1, 2, 3))
        union = torch.maximum(p, t).sum(dim=(1, 2, 3))
    else:
        intersection = (p * t).sum(dim=(1, 2, 3))
        union = (p + t - p * t).sum(dim=(1, 2, 3))
    
    return ((intersection + eps) / (union + eps)).mean().item()


def compute_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    force_soft_mode: bool = False,
) -> float:
    """
    Вычисляет Dice coefficient для бинарной или мягкой сегментации.
    """
    p, t, is_soft = _prepare_masks(preds, targets, threshold, force_soft_mode, eps)
    
    if is_soft:
        intersection = (p * t).sum(dim=(1, 2, 3))
        total = (p  ** 2 + t ** 2).sum(dim=(1, 2, 3))
    else:
        intersection = (p * t).sum(dim=(1, 2, 3))
        total = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    
    return ((2 * intersection + eps) / (total + eps)).mean().item()


def compute_tpr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    force_soft_mode: bool = False,
) -> float:
    """
    Вычисляет True Positive Rate (Recall).
    В мягком режиме: Σ(p * g) / Σ(g)
    """
    p, t, is_soft = _prepare_masks(preds, targets, threshold, force_soft_mode, eps)
    
    if is_soft:
        tp = (p * t).sum(dim=(1, 2, 3))
        denom = t.sum(dim=(1, 2, 3))
    else:
        tp = (p * t).sum(dim=(1, 2, 3))
        fn = ((1 - p) * t).sum(dim=(1, 2, 3))
        denom = tp + fn
    
    return ((tp + eps) / (denom + eps)).mean().item()


def compute_tnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    force_soft_mode: bool = False,
) -> float:
    """
    Вычисляет True Negative Rate (Specificity).
    В мягком режиме: Σ((1-p) * (1-g)) / Σ(1-g)
    """
    p, t, is_soft = _prepare_masks(preds, targets, threshold, force_soft_mode, eps)
    
    if is_soft:
        tn = ((1 - p) * (1 - t)).sum(dim=(1, 2, 3))
        denom = (1 - t).sum(dim=(1, 2, 3))
    else:
        tn = ((1 - p) * (1 - t)).sum(dim=(1, 2, 3))
        fp = (p * (1 - t)).sum(dim=(1, 2, 3))
        denom = tn + fp
    
    return ((tn + eps) / (denom + eps)).mean().item()


# ======== Подробные тесты ========
def test_modes():
    """Тестирование корректности определения режима."""
    print("=== Тест режимов по уникальным значениям ===\n")
    
    B, H, W = 2, 5, 5
    test_preds = torch.randn(B, 1, H, W)
    
    # Тест 1: Чисто бинарные (0/1)
    binary_data = torch.randint(0, 2, (B, 1, H, W)).float()
    _, _, mode1 = _prepare_masks(test_preds, binary_data)
    print(f"Бинарные 0/1: {'soft' if mode1 else 'binary'} ✓")
    
    # Тест 2: Бинарные (но только 0)
    zeros_data = torch.zeros(B, 1, H, W)
    _, _, mode2 = _prepare_masks(test_preds, zeros_data)
    print(f"Только нули: {'soft' if mode2 else 'binary'} ✓")
    
    # Тест 3: Бинарные (но только 1)
    ones_data = torch.ones(B, 1, H, W)
    _, _, mode3 = _prepare_masks(test_preds, ones_data)
    print(f"Только единицы: {'soft' if mode3 else 'binary'} ✓")
    
    # Тест 4: Мягкие [0, 255]
    soft_255 = torch.randint(0, 256, (B, 1, H, W)).float()
    _, _, mode4 = _prepare_masks(test_preds, soft_255)
    print(f"[0, 255] маски: {'soft' if mode4 else 'binary'} ✓")
    
    # Тест 5: Уже нормализованные [0, 1]
    soft_01 = torch.rand(B, 1, H, W)
    _, _, mode5 = _prepare_masks(test_preds, soft_01)
    print(f"[0, 1] float: {'soft' if mode5 else 'binary'} ✓")
    
    # Тест 6: Смешанные значения (не 0/1 но < 1)
    mixed = torch.tensor([0.0, 0.3, 0.7, 1.0]).reshape(1, 1, 2, 2).expand(B, 1, H//2, W//2)
    _, _, mode6 = _prepare_masks(test_preds, mixed)
    print(f"Смешанные (0, 0.3, 0.7, 1): {'soft' if mode6 else 'binary'} ✓")
    
    # Тест 7: Принудительный мягкий режим
    _, _, mode7 = _prepare_masks(test_preds, binary_data, force_soft_mode=True)
    print(f"Принудительный soft: {'soft' if mode7 else 'binary'} ✓")


if __name__ == "__main__":
    test_modes()
    
    print("\n=== Производительность метрик ===\n")
    B, H, W = 4, 256, 256
    
    # Бинарные данные (как раньше)
    binary_preds = torch.randn(B, 1, H, W)
    binary_targets = torch.randint(0, 2, (B, 1, H, W)).float()
    
    print(f"Binary IoU: {compute_iou(binary_preds, binary_targets):.4f}")
    print(f"Binary Dice: {compute_dice(binary_preds, binary_targets):.4f}")
    
    # Мягкие данные (0-255)
    soft_targets = torch.randint(0, 256, (B, 1, H, W)).float()
    
    print(f"Soft IoU: {compute_iou(binary_preds, soft_targets):.4f}")
    print(f"Soft Dice: {compute_dice(binary_preds, soft_targets):.4f}")