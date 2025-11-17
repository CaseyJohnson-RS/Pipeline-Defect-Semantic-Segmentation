import torch
from tqdm import tqdm
from src.models.scoring import compute_iou, compute_tpr, compute_tnr
from src.console import gradient_color


def _detect_mask_type(masks: torch.Tensor, eps: float = 1e-6) -> bool:
    """
    Определяет тип масок по первому батчу.
    Возвращает True если это soft masks (карты вероятностей).
    """
    # Быстрая проверка: если max > 1, то это точно soft
    if masks.max() > 1.0 + eps:
        return True
    
    # Проверяем уникальные значения
    unique_vals = torch.unique(masks)
    is_binary = torch.all(
        torch.logical_or(
            torch.abs(unique_vals) < eps,
            torch.abs(unique_vals - 1.0) < eps
        )
    )
    
    return not is_binary


def semantic_segmentation_evaluation(model, val_loader, criterion, device, log=False, prefix="Validation", colour="yellow"):
    model.to(device)
    model.eval()
    
    # Определяем тип масок на первом батче
    first_batch = next(iter(val_loader))
    _, sample_masks = first_batch
    is_soft_mode = _detect_mask_type(sample_masks)
    
    # Соответствующие названия метрик
    metric_names = {
        'iou': 'Soft IoU' if is_soft_mode else 'IoU',
        'dice': 'Soft Dice' if is_soft_mode else 'Dice',
        'recall': 'Soft Recall' if is_soft_mode else 'Recall',
        'specificity': 'Soft Specificity' if is_soft_mode else 'Specificity'
    }
    
    total_iou = 0.0
    total_loss = 0.0
    total_recall = 0.0
    total_specificity = 0.0

    if log:
        val_loader = tqdm(val_loader, desc=prefix, leave=False, position=1, colour=colour)

    steps = 0
    with torch.no_grad():
        for images, masks in val_loader:
            steps += 1

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            total_iou += compute_iou(outputs, masks)
            total_recall += compute_tpr(outputs, masks)
            total_specificity += compute_tnr(outputs, masks)
            total_loss += criterion(outputs, masks).item()
            
            if log:
                val_loader.set_postfix({'loss': total_loss / steps})

    avg_iou = total_iou / len(val_loader)
    avg_dice = (2 * avg_iou) / (avg_iou + 1)
    avg_loss = total_loss / len(val_loader)
    avg_recall = total_recall / len(val_loader)
    avg_specificity = total_specificity / len(val_loader)

    # Формируем выход с корректными названиями
    out = {
        'metrics': { 
            metric_names['iou']: avg_iou, 
            metric_names['dice']: avg_dice, 
            'Loss': avg_loss, 
            metric_names['recall']: avg_recall, 
            metric_names['specificity']: avg_specificity,
        },
        'console_log': (
            f'Loss:\t{round(avg_loss, 3)}\t' +
            f'{metric_names["iou"]}:\t{gradient_color(round(avg_iou, 3))}\t' +
            f'{metric_names["dice"]}:\t{gradient_color(round(avg_dice, 3))}\t' +
            f'{metric_names["recall"]}:\t{gradient_color(round(avg_recall, 3))}\t' +
            f'{metric_names["specificity"]}:\t{gradient_color(round(avg_specificity, 3))}'
        )
    }

    return out