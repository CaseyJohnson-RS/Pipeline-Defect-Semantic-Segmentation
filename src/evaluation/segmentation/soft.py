import torch
from tqdm import tqdm
from src.scoring.segmetrics.soft import iou, tpr, tnr   # segmetrics.soft
from src.console import gradient_color


def evaluate_soft(
        model, val_loader, criterion, device,
        *, log: bool = True, prefix: str = "Validation (soft)",
        colour: str = "yellow"):
    """
    Evaluate the model in soft mode (targets are probability maps).
    Returns a dict with keys 'metrics' and 'console_log'.
    """
    model.to(device)
    model.eval()

    total_iou = total_loss = total_rec = total_spec = 0.0
    steps = 0

    loader = tqdm(val_loader, desc=prefix, leave=False, position=1, colour=colour) if log else val_loader

    with torch.no_grad():
        for images, masks in loader:
            steps += 1
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            total_iou   += iou(outputs, masks)
            total_rec   += tpr(outputs, masks)
            total_spec  += tnr(outputs, masks)
            total_loss  += criterion(outputs, masks).item()

            if log:
                loader.set_postfix({'loss': total_loss / steps})

    n = len(val_loader)
    avg_iou  = total_iou / n
    avg_dice = (2 * avg_iou) / (avg_iou + 1)
    avg_loss = total_loss / n
    avg_rec  = total_rec / n
    avg_spec = total_spec / n

    metrics = {
        'Soft IoU': avg_iou,
        'Soft Dice': avg_dice,
        'Loss': avg_loss,
        'Soft Recall': avg_rec,
        'Soft Specificity': avg_spec,
    }
    console = (
        f'Loss:\t{avg_loss:.3f}\t'
        f'Soft IoU:\t{gradient_color(avg_iou)}\t'
        f'Soft Dice:\t{gradient_color(avg_dice)}\t'
        f'Soft Recall:\t{gradient_color(avg_rec)}\t'
        f'Soft Specificity:\t{gradient_color(avg_spec)}'
    )
    return {'metrics': metrics, 'console_log': console}