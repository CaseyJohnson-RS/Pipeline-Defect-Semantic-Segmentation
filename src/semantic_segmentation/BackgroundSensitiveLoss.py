import torch
import torch.nn as nn

class BackgroundSensitiveLoss(nn.Module):
    """
    Кастомная функция потерь для бинарной семантической сегментации.
    Ошибки на фоне штрафуются сильнее, чем ошибки на объекте.
    Работает с масками 0/1 и логитами.
    """
    def __init__(self, background_weight: float = 1.0, object_weight: float = 0.9):
        super().__init__()
        self.background_weight = background_weight
        self.object_weight = object_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W) — выход модели до сигмоиды
        targets: (B, 1, H, W) — бинарная маска 0/1
        """
        # Веса пикселей
        weights = self.background_weight + (self.object_weight - self.background_weight) * targets
        # BCE с логитами и весами
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), weight=weights, reduction='mean')
        return bce
