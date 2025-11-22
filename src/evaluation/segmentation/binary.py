import torch
from tqdm import tqdm
from src.scoring.segmetrics.binary import iou, tpr, tnr   # segmetrics.binary
from src.console import gradient_color


def evaluate_binary(
        model, val_loader, criterion, device, *,
        threshold: float = 0.5,
        log: bool = True, prefix: str = "Validation (binary)",
        colour: str = "yellow"):
    """
    Evaluate the model in binary mode.
    threshold -- binarization threshold for logits/probabilities.
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

            total_iou   += iou(outputs, masks, threshold=threshold)
            total_rec   += tpr(outputs, masks, threshold=threshold)
            total_spec  += tnr(outputs, masks, threshold=threshold)
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
        'IoU': avg_iou,
        'Dice': avg_dice,
        'Loss': avg_loss,
        'Recall': avg_rec,
        'Specificity': avg_spec,
    }
    console = (
        f'Loss:\t{avg_loss:.3f}\t'
        f'IoU:\t{gradient_color(avg_iou)}\t'
        f'Dice:\t{gradient_color(avg_dice)}\t'
        f'Recall:\t{gradient_color(avg_rec)}\t'
        f'Specificity:\t{gradient_color(avg_spec)}'
    )
    return {'metrics': metrics, 'console_log': console}