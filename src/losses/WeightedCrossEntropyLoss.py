import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted binary cross-entropy loss for semantic segmentation.

    Assigns different weights to errors on background (class 0) and object (class 1),
    which is useful for class imbalance.

    Args:
        background_weight (float): weight for background (class 0). Default: 0.1.
        object_weight (float): weight for object (class 1). Default: 1.0.

    Shape:
        Input (logits): (B, 1, H, W) — model logits (before sigmoid).
        Target (targets): (B, 1, H, W) — binary masks (0 or 1).
        Output: scalar — loss value (mean reduction).

    Example:
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
        Initialize the loss layer.

        Args:
            background_weight: weight for background pixels (class 0).
            object_weight: weight for object pixels (class 1).
            reduction: loss aggregation method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.background_weight = background_weight
        self.object_weight = object_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted binary cross-entropy.

        Args:
            logits: model logits of shape (B, 1, H, W).
            targets: binary masks of shape (B, 1, H, W) with values 0 or 1.

        Returns:
            Scalar (with reduction='mean' or 'sum') or per-element loss tensor.
        """
        # Check dimension correspondence
        if logits.shape != targets.shape:
            raise ValueError(
                f"Logits shape {logits.shape} and targets shape {targets.shape} do not match."
            )

        # Create weight mask: background_weight for background, object_weight for object
        weights = (
            self.background_weight
            + (self.object_weight - self.background_weight) * targets.float()
        )

        # Compute binary cross-entropy with logits and weights
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), weight=weights, reduction=self.reduction
        )
        return loss