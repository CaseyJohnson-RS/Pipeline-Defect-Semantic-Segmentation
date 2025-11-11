import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(
    model,
    dataset,
    device,
    save_path="predictions.png",
    num_samples=12,
    threshold=0.5,
    overlay_alpha=0.5,
):
    """
    Визуализирует результаты сегментации в красивом формате:
        [Image] | [Ground Truth] | [Overlay (GT=Green, Pred=Red)]

    Args:
        model: torch.nn.Module — обученная модель
        dataset: torch.utils.data.Dataset — набор данных
        device: torch.device — устройство (CPU/GPU)
        save_path: str — путь для сохранения визуализации
        num_samples: int — сколько примеров отобразить
        threshold: float — порог бинаризации предсказаний
        overlay_alpha: float — прозрачность оверлея
    """
    model.eval()
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    # 3 столбца: [Image, GT, Overlay]
    cols = 3
    rows = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    
    # Если num_samples == 1, axes не двумерен, исправим
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Цвета
    gt_color = np.array([0, 255, 0]) / 255.0   # зелёный
    pred_color = np.array([255, 0, 0]) / 255.0 # красный

    for row_idx, img_idx in enumerate(indices):
        image, mask = dataset[img_idx]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        # --- Предсказание ---
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()
            pred_bin = (pred > threshold).astype(np.uint8)

        # --- Изображения ---
        img_orig = np.clip(image_np, 0, 1)

        # Ground Truth
        img_gt = img_orig.copy()
        img_gt[mask_np > 0.5] = (
            gt_color * 0.7 + img_gt[mask_np > 0.5] * (1 - 0.7)
        )

        # Overlay: зелёный GT, красный предикт
        img_overlay = img_orig.copy()
        img_overlay[mask_np > 0.5] = (
            gt_color * overlay_alpha + img_overlay[mask_np > 0.5] * (1 - overlay_alpha)
        )
        img_overlay[pred_bin > 0.5] = (
            pred_color * overlay_alpha + img_overlay[pred_bin > 0.5] * (1 - overlay_alpha)
        )

        # --- Построение ---
        axes[row_idx, 0].imshow(img_orig)
        axes[row_idx, 0].set_title(f"Image {img_idx}", fontsize=10)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(img_gt)
        axes[row_idx, 1].set_title("Ground Truth", fontsize=10)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(img_overlay)
        axes[row_idx, 2].set_title("Overlay (GT=Green, Pred=Red)", fontsize=10)
        axes[row_idx, 2].axis("off")

    # --- Оформление ---
    fig.suptitle("Model Predictions Overview", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved visualization to: {save_path}")
