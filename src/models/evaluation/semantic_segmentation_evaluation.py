import torch
from tqdm import tqdm
from src.models.scoring import compute_dice, compute_iou, compute_tpr, compute_tnr
from src.console import gradient_color

def semantic_segmentation_evaluation(model, val_loader, criterion, device, log=False):
    model.to(device)
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_loss = 0.0
    total_recall = 0.0
    total_specificity = 0.0

    if (log):
        val_loader = tqdm(val_loader, desc="=== Validation ===", leave=False, position=1, colour="yellow")

    steps = 0
    with torch.no_grad():
        for images, masks in val_loader:
            steps += 1

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            total_iou += compute_iou(outputs, masks)
            total_dice += compute_dice(outputs, masks)
            total_recall += compute_tpr(outputs, masks)
            total_specificity ++ compute_tnr(outputs, masks)
            total_loss += criterion(outputs, masks).item()
            
            val_loader.set_postfix({'loss': total_loss / steps })

    avg_iou = total_iou / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_loss = total_loss / len(val_loader)
    avg_recall = total_recall / len(val_loader)
    avg_specificity = total_specificity / len(val_loader)

    out = {
        'metrics': { 
            'IoU': avg_iou, 
            'Dice': avg_dice, 
            'Loss': avg_loss, 
            'Recall': avg_recall, 
            'Specificity': avg_specificity,
        },
        'console_log': (
            f'Loss:\t{round(avg_loss, 3)}\t' +
            f'IoU:\t{gradient_color(round(avg_iou, 3))}\t' +
            f'Dice:\t{gradient_color(round(avg_dice, 3))}\t' +
            f'Recall:\t{gradient_color(round(avg_recall, 3))}\t' +
            f'Specificity:\t{gradient_color(round(avg_specificity, 3))}'
        )

    }

    return out
