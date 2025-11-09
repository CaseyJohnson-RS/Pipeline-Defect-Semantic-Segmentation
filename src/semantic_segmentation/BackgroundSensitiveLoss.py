import torch
import torch.nn as nn

class BackgroundSensitiveLoss(nn.Module):
    """
    Кастомная функция потерь для семантической сегментации.
    Жестко штрафует ошибки на фоне и мягко — на объекте.
    """
    def __init__(self, background_weight=3.0, object_weight=1.0):
        super(BackgroundSensitiveLoss, self).__init__()
        self.background_weight = background_weight
        self.object_weight = object_weight

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W) — выход модели до сигмоиды
        targets: (B, 1, H, W) — бинарная карта (0 — фон, 1 — объект)
        """
        # Применяем сигмоиду к логитам
        probs = torch.sigmoid(logits)

        # Веса для каждого пикселя: больше для фона, меньше для объекта
        weights = torch.where(targets == 0, 
                              torch.full_like(targets, self.background_weight), 
                              torch.full_like(targets, self.object_weight))
        
        # BCE вручную, чтобы контролировать веса
        bce = -(targets * torch.log(probs + 1e-8) + (1 - targets) * torch.log(1 - probs + 1e-8))
        loss = (weights * bce).mean()

        return loss
