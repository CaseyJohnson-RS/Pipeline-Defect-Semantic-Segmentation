import torch
from src.semantic_segmentation.metrics import compute_iou, compute_dice


def test_iou_perfect_match():
    preds = torch.ones((2, 1, 4, 4))
    targets = torch.ones((2, 1, 4, 4))
    result = compute_iou(preds, targets)
    assert abs(result - 1.0) < 1e-6


def test_iou_no_overlap():
    preds = torch.zeros((2, 1, 4, 4))
    targets = torch.ones((2, 1, 4, 4))
    result = compute_iou(preds, targets)
    assert abs(result - 0.0) < 1e-6


def test_iou_half_overlap():
    preds = torch.tensor([
        [[[1, 1, 0, 0],
          [1, 1, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]],
        [[[0, 0, 0, 0],
          [0, 1, 1, 0],
          [0, 1, 1, 0],
          [0, 0, 0, 0]]]
    ]).float()

    targets = torch.tensor([
        [[[1, 1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]],
        [[[0, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]]
    ]).float()

    # Для первой маски IoU = 3/4 = 0.75, для второй 1/4 = 0.25
    result = compute_iou(preds, targets)
    assert abs(result - 0.5) < 1e-6


def test_dice_perfect_match():
    preds = torch.ones((2, 1, 4, 4))
    targets = torch.ones((2, 1, 4, 4))
    result = compute_dice(preds, targets)
    assert abs(result - 1.0) < 1e-6


def test_dice_no_overlap():
    preds = torch.zeros((2, 1, 4, 4))
    targets = torch.ones((2, 1, 4, 4))
    result = compute_dice(preds, targets)
    assert abs(result - 0.0) < 1e-6


def test_dice_half_overlap():
    preds = torch.tensor([
        [[[1, 1, 0, 0],
          [1, 1, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]]
    ]).float()

    targets = torch.tensor([
        [[[1, 1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]]
    ]).float()

    # intersection=3, sum=5 => dice = 2*3/5 = 1.2 = 0.6
    result = compute_dice(preds, targets)
    assert abs(result - 0.6) < 1e-6


def test_logits_input_handling():
    """Проверяем, что функции корректно применяют сигмоиду для логитов"""
    logits = torch.tensor([[[[10.0]], [[-10.0]]]])  # После сигмоиды ~1 и ~0
    targets = torch.tensor([[[[1.0]], [[0.0]]]])

    iou = compute_iou(logits, targets)
    dice = compute_dice(logits, targets)
    assert abs(iou - 1.0) < 1e-6
    assert abs(dice - 1.0) < 1e-6


def test_threshold_effect():
    """Проверка влияния порога"""
    preds = torch.tensor([[[[0.4]], [[0.6]]]])
    targets = torch.tensor([[[[1.0]], [[1.0]]]])

    iou_low = compute_iou(preds, targets, threshold=0.3)
    iou_high = compute_iou(preds, targets, threshold=0.7)

    assert iou_low > iou_high
