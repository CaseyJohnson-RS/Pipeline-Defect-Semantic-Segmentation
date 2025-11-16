import cv2
import numpy as np
import torch
import torch.nn as nn

class BboxAwareLoss(nn.Module):
    """
    BCE Loss с пониженным весом на границах bbox.
    Границы bbox получим через морфологический градиент.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, pred, target):
        # target может быть soft [0.0-1.0]
        loss = self.bce(pred, target)
        
        # Если это "жесткая" маска (0/1), создаем веса
        if target.max() == 1.0 and target.min() == 0.0:
            # Граница bbox (дилатация - эрозия)
            mask_np = target.cpu().numpy()
            weights = np.ones_like(mask_np)
            
            for i in range(mask_np.shape[0]):
                binary_mask = (mask_np[i, 0] > 0.5).astype(np.uint8)
                if binary_mask.sum() > 0:
                    kernel = np.ones((5, 5), np.uint8)
                    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
                    eroded = cv2.erode(binary_mask, kernel, iterations=1)
                    edge = dilated - eroded
                    weights[i, 0][edge > 0] = 0.1  # Низкий вес на границах
            
            weights = torch.from_numpy(weights).to(pred.device)
            loss = loss * weights
        
        return loss.mean()