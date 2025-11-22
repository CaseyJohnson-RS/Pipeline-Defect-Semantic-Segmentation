"""
Explicit (single-mode) segmentation metrics.
Usage:
    from segmetrics import soft, binary
    iou = soft.iou(preds, targets)
    dice = binary.dice(preds, targets, threshold=0.4)
"""
from . import soft, binary  # noqa: F401