import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_segmentation_model(model_path: str):
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    return model

def load_yolo_boxes(label_path: str, img_w: int, img_h: int, target_class: int):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            cls_id, x, y, w, h = map(float, line.split())
            if int(cls_id) != target_class:
                continue
            bx = int((x - w / 2) * img_w)
            by = int((y - h / 2) * img_h)
            bw = int(w * img_w)
            bh = int(h * img_h)
            boxes.append([bx, by, bx + bw, by + bh])
    return boxes

def box_to_mask(box, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 255
    return mask

def predict_mask(model, image):
    # Пример для модели, отдающей бинарную маску через сигмоиду
    with torch.no_grad():
        inp = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        pred = (model(inp)[0].sigmoid().squeeze().cpu().numpy() > 0.45).astype(np.uint8) * 255
    return pred

def process_dataset(
        model_path: str,
        dataset_path: str,
        yolo_labels_path: str,
        output_path: str,
        split: str = "train",
        target_class: int = 0,
        overlap_threshold: float = 0.5
):
    os.makedirs(f"{output_path}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_path}/masks/{split}", exist_ok=True)

    model = load_segmentation_model(model_path)

    img_dir = Path(dataset_path) / "images" / split
    for img_file in img_dir.glob("*.jpg"):
        image = cv2.imread(str(img_file))
        h, w = image.shape[:2]

        pred_mask = predict_mask(model, image)

        label_file = Path(yolo_labels_path) / f"{img_file.stem}.txt"
        boxes = load_yolo_boxes(str(label_file), w, h, target_class)

        final_mask = np.zeros((h, w), dtype=np.uint8)

        for box in boxes:
            box_mask = box_to_mask(box, (h, w))

            intersection = cv2.bitwise_and(pred_mask, box_mask)
            inter_area = np.count_nonzero(intersection)
            box_area = np.count_nonzero(box_mask)
            overlap = inter_area / box_area if box_area > 0 else 0

            if overlap < overlap_threshold:
                final_mask = cv2.bitwise_or(final_mask, box_mask)
            else:
                final_mask = cv2.bitwise_or(final_mask, intersection)

        out_img = f"{output_path}/images/{split}/{img_file.name}"
        out_mask = f"{output_path}/masks/{split}/{img_file.stem}.png"

        cv2.imwrite(out_img, image)
        cv2.imwrite(out_mask, final_mask)

        print(f"Processed {img_file.name}")

# Пример вызова:
process_dataset(
    model_path="models/unet_bss_Deformation_augmented_Test_Loss_Params_1x3.pth",
    dataset_path="datasets/Deformation",
    yolo_labels_path="datasets/tmp",
    output_path="datasets/Deformation_v1",
    split="val",
    target_class=0,
    overlap_threshold=0.4
)
