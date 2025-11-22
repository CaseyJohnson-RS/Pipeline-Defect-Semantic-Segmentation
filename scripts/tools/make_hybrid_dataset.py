import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import torch
from torch.utils.data import Dataset
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.models.factories import load_unet_attention  # noqa: E402

# ============== CONFIGURATION ==============
DATASET_NAME = input("Enter dataset name: ")
SOURCE_DIR = Path(f"datasets/{DATASET_NAME}")
BASELINE_DIR = Path(f"datasets/{DATASET_NAME}_BASELINE")
OUTPUT_DIR = Path(f"datasets/{DATASET_NAME}_H")  # Will be updated with alpha value

# Model and mask enhancement parameters
ALPHA = 0.2  # Coefficient for hybrid masks
THRESHOLD = 0.8 # Coefficient for model's confidence
MODEL_INPUT_SIZE = (702, 512)  # Model input size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Augmentation parameters
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.85  # 85% for train
NUM_AUGMENTATIONS = 3   # Number of augmented images per original image
# ============================================

class ImageMaskDataset(Dataset):
    """Dataset for loading images and masks"""
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask, self.image_paths[idx]

def parse_polygon(annotation_str):
    """Parses JSON string with polygon and returns coordinates"""
    try:
        # === ADDED CHECK: Skip empty strings ===
        if not annotation_str or annotation_str.strip() in ['{}', '""{}""', '""', '"" ""']:
            return None, None
            
        cleaned = annotation_str.replace('""', '"')
        
        # === ADDED CHECK: Re-check after cleaning ===
        if not cleaned or cleaned.strip() == '{}':
            return None, None
            
        data = json.loads(cleaned)
        # === ADDED CHECK: Check for required fields ===
        if data.get('name') == 'polygon' and 'all_points_x' in data and 'all_points_y' in data:
            return data['all_points_x'], data['all_points_y']
        return None, None
    except Exception as e:
        print(f"Polygon parsing error: {e}, line: {annotation_str[:50]}...")
        return None, None

def create_mask_from_polygon(image_shape, points_x, points_y):
    """Creates binary mask from polygon coordinates"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array(list(zip(points_x, points_y)), dtype=np.int32)
    cv2.fillPoly(mask, [points], color=255)
    return mask

def copy_or_create_dirs(*dirs):
    """Creates directories if they don't exist"""
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_file_mapping(directory):
    """Creates dict {name_without_extension: full_filename}"""
    mapping = {}
    for file_path in directory.iterdir():
        if file_path.is_file():
            name_without_ext = file_path.stem
            mapping[name_without_ext] = file_path.name
    return mapping

def get_pixel_perfect_names(labels_path):
    """Gets list of filenames with pixel-perfect masks"""
    df = pd.read_csv(labels_path)
    
    # === ADDED CHECK: Filter files without annotations ===
    initial_count = len(df)
    df_valid = df[
        (df['region_count'] > 0) & 
        (df['region_shape_attributes'].notna()) & 
        (~df['region_shape_attributes'].astype(str).str.strip().isin(['{}', '""{}""', '']))
    ]
    
    filtered_count = initial_count - len(df_valid)
    if filtered_count > 0:
        print(f"✅ Filtered {filtered_count} records without valid annotations")
    
    if len(df_valid) == 0:
        print("⚠️  Warning: No files with valid pixel-perfect annotations found!")
        
    pixel_perfect_names = set(Path(filename).stem for filename in df_valid['filename'])
    print(f"✅ Found {len(pixel_perfect_names)} unique files with pixel-perfect masks")
    return pixel_perfect_names

def preprocess_for_model(image, target_size):
    """Prepares image for model"""
    # Resize
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    # Normalize
    image_norm = image_resized.astype(np.float32) / 255.0
    # Convert to tensor format: (C, H, W)
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)
    return image_tensor

def predict_mask(model, image_path, target_size, device):
    """Gets mask prediction from model"""
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    # Prepare for model
    image_tensor = preprocess_for_model(image, target_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Prediction
    with torch.no_grad():
        prediction = model.predict(image_tensor)
        
        # === FIX: Convert tensor to NumPy ===
        # Handle different model output formats
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu()
        
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.numpy()
        # =====================================
        
        # Now prediction is a numpy array
        # Assume prediction has shape (1, H, W) or (1, 1, H, W) or (H, W)
        if prediction.ndim == 4:
            # (B, C, H, W) -> take first channel of first batch
            prediction = prediction[0, 0]
        elif prediction.ndim == 3:
            # (B, H, W) -> take first batch
            prediction = prediction[0]
        # If ndim == 2, already (H, W)
        
        # Ensure values in range [0, 1]
        prediction = np.clip(prediction, 0, 1)
        
    # Resize back to original size
    pred_mask = (prediction * 255).astype(np.uint8)
    pred_mask_resized = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
    return pred_mask_resized

def augment_image_and_mask(image, mask, seed=None):
    """Applies random light augmentation to image and mask"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    augmented_image = image.copy()
    augmented_mask = mask.copy()
    
    aug_type = random.choice(['hflip', 'vflip', 'rotate'])
    
    if aug_type == 'hflip':
        augmented_image = cv2.flip(augmented_image, 1)
        augmented_mask = cv2.flip(augmented_mask, 1)
        
    elif aug_type == 'vflip':
        augmented_image = cv2.flip(augmented_image, 0)
        augmented_mask = cv2.flip(augmented_mask, 0)
        
    elif aug_type == 'rotate':
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented_image = cv2.warpAffine(augmented_image, M, (w, h))
        augmented_mask = cv2.warpAffine(augmented_mask, M, (w, h))
    
    return augmented_image, augmented_mask, aug_type

def main():
    global images_dir, image_mapping
    
    # 1. Scan images folder
    images_dir = SOURCE_DIR / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Folder {images_dir} not found!")
    
    image_mapping = get_file_mapping(images_dir)
    print(f"Found {len(image_mapping)} files in images folder")
    
    # 2. Get list of files with pixel-perfect masks
    labels_path = SOURCE_DIR / "labels.csv"
    # === CHANGED: Function now automatically filters files without annotations ===
    pixel_perfect_names = get_pixel_perfect_names(labels_path)
    
    # 3. Determine files without pixel-perfect masks (for improvement)
    all_names = set(image_mapping.keys())
    names_to_improve = list(all_names - pixel_perfect_names)
    print(f"Found {len(names_to_improve)} files to improve with model")
    
    # 4. Load model
    print("\nLoading model...")
    model = load_unet_attention()
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    dataset_num = int(input("Enter dataset number: "))

    # 5. Create output directory with alpha
    output_dir = Path(f"./datasets/{DATASET_NAME}_H{dataset_num}")
    output_images_train = output_dir / "images" / "train"
    output_images_val = output_dir / "images" / "val"
    output_masks_train = output_dir / "masks" / "train"
    output_masks_val = output_dir / "masks" / "val"
    
    copy_or_create_dirs(output_images_train, output_images_val, 
                       output_masks_train, output_masks_val)
    
    # 6. Process files for mask improvement
    train_files, val_files = train_test_split(
        names_to_improve,
        train_size=TRAIN_VAL_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"\nProcessing {len(train_files)} train files and {len(val_files)} val files...")
    
    def process_and_save(name_list, img_dest, mask_dest, split_name):
        """Processes files, creates hybrid masks and saves"""
        print(f"\n{split_name}: creating hybrid masks...")
        
        for name in name_list:
            real_filename = image_mapping[name]
            src_image_path = images_dir / real_filename
            src_mask_path = SOURCE_DIR / "masks" / real_filename
            
            # Check file existence
            if not src_image_path.exists():
                print(f"Warning: Image {src_image_path} not found")
                continue
            if not src_mask_path.exists():
                print(f"Warning: Mask {src_mask_path} not found")
                continue
            
            # Load image and original mask
            image = cv2.imread(str(src_image_path))
            original_mask = cv2.imread(str(src_mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or original_mask is None:
                print(f"Error reading files for {name}")
                continue
            
            # Get model prediction
            try:
                predicted_mask = predict_mask(model, src_image_path, MODEL_INPUT_SIZE, DEVICE)
                predicted_mask[original_mask == 0] = 0
            except Exception as e:
                print(f"Prediction error for {name}: {e}")
                # If error, use only original mask
                predicted_mask = np.zeros_like(original_mask)
            
            # Create hybrid mask
            hybrid_mask = cv2.addWeighted(
                original_mask, ALPHA,
                predicted_mask, (1 - ALPHA),
                0
            )
            
            # Save image and hybrid mask
            dest_image_path = img_dest / real_filename
            mask_filename = Path(real_filename).stem + ".png"
            dest_mask_path = mask_dest / mask_filename
            
            try:
                cv2.imwrite(str(dest_image_path), image)
                cv2.imwrite(str(dest_mask_path), hybrid_mask)
            except Exception as e:
                print(f"Save error {name}: {e}")
                continue
            
            if len(name_list) <= 5:
                print(f"  {name}: hybrid mask created (alpha={ALPHA})")
    
    # Create hybrid masks
    process_and_save(train_files, output_images_train, output_masks_train, "Hybrid train")
    process_and_save(val_files, output_images_val, output_masks_val, "Hybrid val")
    
    # 7. Add non-augmented images from BASELINE
    print("\nAdding originals from BASELINE...")
    
    baseline_train_dir = BASELINE_DIR / "images" / "train"
    baseline_train_files = [f for f in baseline_train_dir.iterdir() 
                           if f.is_file() and "_aug" not in f.name]
    
    for image_file in baseline_train_files:
        # Copy image
        src_image = image_file
        dest_image = output_images_train / image_file.name
        if not dest_image.exists():
            image = cv2.imread(str(src_image))
            cv2.imwrite(str(dest_image), image)
        
        # Copy mask
        mask_name = image_file.stem + ".png"
        src_mask = BASELINE_DIR / "masks" / "train" / mask_name
        dest_mask = output_masks_train / mask_name
        
        if src_mask.exists() and not dest_mask.exists():
            mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(dest_mask), mask)
    
    # Add validation files from BASELINE
    baseline_val_dir = BASELINE_DIR / "images" / "val"
    baseline_val_files = [f for f in baseline_val_dir.iterdir() if f.is_file()]
    
    for image_file in baseline_val_files:
        # Copy image
        src_image = image_file
        dest_image = output_images_val / image_file.name
        if not dest_image.exists():
            image = cv2.imread(str(src_image))
            cv2.imwrite(str(dest_image), image)
        
        # Copy mask
        mask_name = image_file.stem + ".png"
        src_mask = BASELINE_DIR / "masks" / "val" / mask_name
        dest_mask = output_masks_val / mask_name
        
        if src_mask.exists() and not dest_mask.exists():
            mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(dest_mask), mask)
    
    print(f"Added {len(baseline_train_files)} train and {len(baseline_val_files)} val files from BASELINE")
    
    # 8. Augmentation of merged dataset (train only)
    def apply_final_augmentation(img_dir, mask_dir):
        """Applies augmentation to all files in directory"""
        print(f"\nApplying final augmentation to {img_dir}...")
        
        image_files = [f for f in img_dir.iterdir() if f.is_file() and "_aug" not in f.name]
        
        for image_file in image_files:
            image = cv2.imread(str(image_file))
            mask_file = mask_dir / (image_file.stem + ".png")
            
            if not mask_file.exists():
                continue
            
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            for i in range(NUM_AUGMENTATIONS):
                aug_seed = RANDOM_SEED + hash(image_file.stem) % 1000 + i
                
                aug_image, aug_mask, aug_type = augment_image_and_mask(
                    image, mask, seed=aug_seed
                )
                
                # Save augmented versions
                aug_image_name = f"{image_file.stem}_aug{i}{image_file.suffix}"
                aug_mask_name = f"{image_file.stem}_aug{i}.png"
                
                cv2.imwrite(str(img_dir / aug_image_name), aug_image)
                cv2.imwrite(str(mask_dir / aug_mask_name), aug_mask)
    
    apply_final_augmentation(output_images_train, output_masks_train)
    
    # 9. Print statistics
    print("\n✅ Done! Dataset created:")
    print(f"   - Path: {output_dir}")
    print(f"   - Alpha: {ALPHA}")
    
    train_images = list(output_images_train.iterdir())
    val_images = list(output_images_val.iterdir())
    total_images = len(train_images) + len(val_images)
    
    print("\n📊 Statistics:")
    print(f"   - Train: {len(train_images)} images")
    print(f"   - Val:   {len(val_images)} images")
    print(f"   - Total: {total_images} images")

if __name__ == "__main__":
    main()