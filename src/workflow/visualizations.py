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
    Visualizes segmentation results in a nice format:
        [Image] | [Ground Truth] | [Overlay (GT=Green, Pred=Red)]

    Args:
        model: torch.nn.Module — trained model
        dataset: torch.utils.data.Dataset — dataset
        device: torch.device — device (CPU/GPU)
        save_path: str — path to save visualization
        num_samples: int — number of examples to display
        threshold: float — threshold for prediction binarization
        overlay_alpha: float — overlay transparency
    """
    model.eval()
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    # 3 columns: [Image, GT, Overlay]
    cols = 3
    rows = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    
    # If num_samples == 1, axes is not 2D, fix it
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Colors
    gt_color = np.array([0, 255, 0]) / 255.0   # green
    pred_color = np.array([255, 0, 0]) / 255.0 # red

    for row_idx, img_idx in enumerate(indices):
        image, mask = dataset[img_idx]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        # --- Prediction ---
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()
            pred_bin = (pred > threshold).astype(np.uint8)

        # --- Images ---
        img_orig = np.clip(image_np, 0, 1)

        # Ground Truth
        img_gt = img_orig.copy()
        img_gt[mask_np > 0.5] = (
            gt_color * 0.7 + img_gt[mask_np > 0.5] * (1 - 0.7)
        )

        # Overlay: green GT, red prediction
        img_overlay = img_orig.copy()
        img_overlay[mask_np > 0.5] = (
            gt_color * overlay_alpha + img_overlay[mask_np > 0.5] * (1 - overlay_alpha)
        )
        img_overlay[pred_bin > 0.5] = (
            pred_color * overlay_alpha + img_overlay[pred_bin > 0.5] * (1 - overlay_alpha)
        )

        # --- Plot ---
        axes[row_idx, 0].imshow(img_orig)
        axes[row_idx, 0].set_title(f"Image {img_idx}", fontsize=10)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(img_gt)
        axes[row_idx, 1].set_title("Ground Truth", fontsize=10)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(img_overlay)
        axes[row_idx, 2].set_title("Overlay (GT=Green, Pred=Red)", fontsize=10)
        axes[row_idx, 2].axis("off")

    # --- Formatting ---
    fig.suptitle("Model Predictions Overview", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved visualization to: {save_path}")
