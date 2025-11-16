import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    """
    Взвешенная бинарная кросс‑энтропия для семантической сегментации.

    Присваивает разные веса ошибкам на фоне (класс 0) и объекте (класс 1),
    что полезно при дисбалансе классов.

    Args:
        background_weight (float): вес для фона (класс 0). По умолчанию 0.1.
        object_weight (float): вес для объекта (класс 1). По умолчанию 1.0.

    Shape:
        Input (logits): (B, 1, H, W) — логиты модели (до сигмоиды).
        Target (targets): (B, 1, H, W) — бинарные маски (0 или 1).
        Output: скаляр — значение потери (mean reduction).

    Пример:
        criterion = WeightedCrossEntropyLoss(background_weight=0.1, object_weight=1.0)
        loss = criterion(logits, masks)
    """

    def __init__(
        self,
        background_weight: float = 0.1,
        object_weight: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Инициализация слоя потери.

        Args:
            background_weight: вес для пикселей фона (класс 0).
            object_weight: вес для пикселей объекта (класс 1).
            reduction: способ агрегирования потерь ('mean', 'sum' или 'none').
        """
        super().__init__()
        self.background_weight = background_weight
        self.object_weight = object_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычисление взвешенной бинарной кросс‑энтропии.

        Args:
            logits: логиты модели формы (B, 1, H, W).
            targets: бинарные маски формы (B, 1, H, W) с значениями 0 или 1.

        Returns:
            Скаляр (при reduction='mean' или 'sum') либо тензор потерь по элементам.
        """
        # Проверяем соответствие размерностей
        if logits.shape != targets.shape:
            raise ValueError(
                f"Размеры logits {logits.shape} и targets {targets.shape} не совпадают."
            )

        # Формируем весовую маску: для фона — background_weight, для объекта — object_weight
        weights = (
            self.background_weight
            + (self.object_weight - self.background_weight) * targets.float()
        )

        # Вычисляем бинарную кросс‑энтропию с логитами и весами
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), weight=weights, reduction=self.reduction
        )
        return loss