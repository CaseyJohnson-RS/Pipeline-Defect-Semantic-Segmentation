
from src.models.scoring import compute_dice, compute_iou
import torch
from tqdm import tqdm

def semantic_segmentation_evaluation(model, val_loader, criterion, device, log=False):
    model.to(device)
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_loss = 0.0

    if (log):
        val_loader = tqdm(val_loader, desc=" == Evaluation ==", leave=False, position=1, colour="yellow")

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            total_iou += compute_iou(outputs, masks)
            total_dice += compute_dice(outputs, masks)
            total_loss += criterion(outputs, masks).item()


    avg_iou = total_iou / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_loss = total_loss / len(val_loader)

    return { 'IoU': avg_iou, 'Dice': avg_dice, 'Loss': avg_loss }
