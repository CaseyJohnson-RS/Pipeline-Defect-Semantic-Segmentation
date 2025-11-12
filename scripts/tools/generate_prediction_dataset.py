import os
import sys
import torch
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
from tqdm import tqdm

# === НАСТРОЙКИ ===

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Путь к исходному датасету
SOURCE_DATASET = "datasets/PipeBoxSegmentation"
MODEL_PATH = "models/unet_bss_v1.pth"
OUTPUT_DATASET = "datasets/PipeSegmentation_v0"

THRESHOLD = 0.425
MIN_PERCENT_INSIDE_MASK = 0.5  # если меньше этого процента, порог понижается циклически
THRESHOLD_STEP = 0.02          # шаг уменьшения порога
MIN_THRESHOLD = 0.05           # нижний предел порога, чтобы не уйти в ноль


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def load_model(path: str):
    """Загружает модель семантической сегментации"""
    model = torch.load(path, map_location=DEVICE, weights_only=False)
    model.eval()
    return model


def prepare_dirs():
    """Создает выходные папки"""
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DATASET}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DATASET}/masks/{split}", exist_ok=True)


def predict_mask(model, image_path, threshold=THRESHOLD):
    """Делает предсказание маски для одного изображения"""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        binary_mask = (probs > threshold).float()

    # убираем batch и channel, оставляем HxW
    return binary_mask.squeeze().cpu(), image


def process_split(model, split):
    """Обрабатывает train/val"""
    img_dir = os.path.join(SOURCE_DATASET, "images", split)
    mask_dir = os.path.join(SOURCE_DATASET, "masks", split)

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Пропускаю {split}: не найдены {img_dir} или {mask_dir}")
        return

    image_files = sorted(os.listdir(img_dir))
    for img_name in tqdm(image_files, desc=f"Processing {split}"):
        img_path = os.path.join(img_dir, img_name)
        base_name, _ = os.path.splitext(img_name)
        mask_name = f"{base_name}_mask.png"
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"⚠️ Warning: mask not found for {img_name} → ожидается {mask_name}")
            continue

        # Загружаем исходную маску (0 — фон, 255 — объект)
        target_mask = read_image(mask_path)[0].float() / 255.0

        # Делаем предсказание
        threshold = THRESHOLD
        pred_mask, image = predict_mask(model, img_path, threshold)
        pred_mask = pred_mask * target_mask

        inside_percent = (pred_mask.sum() / (target_mask.sum() + 1e-6)).item()

        # Понижаем порог, если процент меньше MIN_PERCENT_INSIDE_MASK
        while inside_percent < MIN_PERCENT_INSIDE_MASK and threshold > MIN_THRESHOLD:
            threshold = max(threshold - THRESHOLD_STEP, MIN_THRESHOLD)
            pred_mask, _ = predict_mask(model, img_path, threshold)
            pred_mask = pred_mask * target_mask
            inside_percent = (pred_mask.sum() / (target_mask.sum() + 1e-6)).item()

        # Сохраняем результат
        image.save(f"{OUTPUT_DATASET}/images/{split}/{img_name}")
        pred_img = Image.fromarray((pred_mask.numpy() * 255).astype("uint8"))
        pred_img.save(f"{OUTPUT_DATASET}/masks/{split}/{mask_name}")


def main():
    print(f"Использую исходный датасет: {os.path.abspath(SOURCE_DATASET)}")
    model = load_model(MODEL_PATH)
    prepare_dirs()

    for split in ["train", "val"]:
        process_split(model, split)

    print("✅ Готово: новые датасеты сохранены.")


if __name__ == "__main__":
    main()
