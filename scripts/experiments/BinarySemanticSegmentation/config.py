from dotenv import load_dotenv
import os
import torch
from src.semantic_segmentation import BackgroundSensitiveLoss  # noqa: F401


load_dotenv()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Общие гиперпараметры
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

SEED = 42

# --- ЛУЧШЕ НЕ ТРОГАТЬ ---

IMAGE_SIZE = (700, 500)
IN_CHANNELS = 3

CLASSES = 1
MODEL_ENCODER_NAME = 'resnet34'
MODEL_ENCODER_WEIGHTS = 'imagenet'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PBSDS_DSA_PATH = "datasets/PipeBoxSegmentation_augmented"
PSDS_DSA_PATH = "datasets/PipeSegmentation_augmented"
PBSDS_DS_PATH = "datasets/PipeBoxSegmentation"
PSDS_DS_PATH = "datasets/PipeSegmentation"



MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')

EVAL_FREQUENCY = 5
VISUALIZATION_SAMPLES = 20


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Гиперпараметры обучения на датасете PBS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

PBS_EPOCHS = 20
PBS_LEARNING_RATE = 1e-4
PBS_BATCH_SIZE = 4
PBS_OPTIMIZER = torch.optim.Adam
PBS_CRITETION = BackgroundSensitiveLoss

# --- ЛУЧШЕ НЕ ТРОГАТЬ ---

PBS_TRAIN_DATASET_PATH = {
    'images': os.path.join(PBSDS_DSA_PATH, 'images', 'train'), 
    'masks': os.path.join(PBSDS_DSA_PATH, 'masks', 'train'), 
}

PBS_VAL_DATASETS = [
    {
        'images': os.path.join(PBSDS_DS_PATH, 'images', 'val'), 
        'masks': os.path.join(PBSDS_DS_PATH, 'masks', 'val'),
        'name': 'PBS Val'
    },
    {
        'images': os.path.join(PSDS_DS_PATH, 'images', 'train'), 
        'masks': os.path.join(PSDS_DS_PATH, 'masks', 'train'),
        'name': 'PS Train'
    }
]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Гиперпараметры обучения на датасете PS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

PS_EPOCHS = 5
PS_LEARNING_RATE = 1e-4
PS_BATCH_SIZE = 2
PS_OPTIMIZER = torch.optim.Adam
PS_CRITETION = BackgroundSensitiveLoss

# --- ЛУЧШЕ НЕ ТРОГАТЬ ---

PS_TRAIN_DATASET_PATH = {
    'images': os.path.join(PSDS_DSA_PATH, 'images', 'train'), 
    'masks': os.path.join(PSDS_DSA_PATH, 'masks', 'train'), 
}


PS_VAL_DATASETS = [
    {
        'images': os.path.join(PSDS_DS_PATH, 'images', 'val'), 
        'masks': os.path.join(PSDS_DS_PATH, 'masks', 'val'), 
        'name': 'PS Val'
    }
]


# ===================================================================================================
# DANGER ZONE
# ===================================================================================================


UNET_MODEL_CONFIG = {
    'encoder_name': MODEL_ENCODER_NAME,
    'in_channels': IN_CHANNELS,
    'classes': CLASSES,
    'device': DEVICE,
    'default_encoder_weights': MODEL_ENCODER_WEIGHTS
}

PBS_TRAIN_CONFIG = {
    'seed': SEED,
    'dataset': 'PBS',
    'epochs': PBS_EPOCHS,
    'eval_frequency': EVAL_FREQUENCY,
    'lr': PBS_LEARNING_RATE,
    'batch_size': PBS_BATCH_SIZE,
    'optimizer': PBS_OPTIMIZER,
    'criterion': PBS_CRITETION,
    'device': DEVICE,
    'train_ds': PBS_TRAIN_DATASET_PATH,
    'val_dss': PBS_VAL_DATASETS,
    'image_size': IMAGE_SIZE,
    'visualization_samples': VISUALIZATION_SAMPLES,
    'log_params': {
        'seed': SEED,
        'dataset': 'PBS',
        'epochs': PBS_EPOCHS,
        'lr': PBS_LEARNING_RATE,
        'batch_size': PBS_BATCH_SIZE,
        'optimizer': PBS_OPTIMIZER.__name__,
        'criterion': PBS_CRITETION.__name__,
        'image_size': IMAGE_SIZE,
        'encoder_name': MODEL_ENCODER_NAME,
        'in_channels': IN_CHANNELS,
        'classes': CLASSES,
        'device': DEVICE,
        'default_encoder_weights': MODEL_ENCODER_WEIGHTS,
        'eval_frequency': EVAL_FREQUENCY,
        'visualization_samples': VISUALIZATION_SAMPLES,
    }
}

PS_TRAIN_CONFIG = {
    'seed': SEED,
    'dataset': 'PS',
    'epochs': PS_EPOCHS,
    'eval_frequency': EVAL_FREQUENCY,
    'lr': PS_LEARNING_RATE,
    'batch_size': PS_BATCH_SIZE,
    'optimizer': PS_OPTIMIZER,
    'criterion': PS_CRITETION,
    'device': DEVICE,
    'train_ds': PS_TRAIN_DATASET_PATH,
    'val_dss': PS_VAL_DATASETS,
    'image_size': IMAGE_SIZE,
    'visualization_samples': VISUALIZATION_SAMPLES,
    'log_params': {
        'seed': SEED,
        'dataset': 'PS',
        'epochs': PS_EPOCHS,
        'lr': PS_LEARNING_RATE,
        'batch_size': PS_BATCH_SIZE,
        'optimizer': PS_OPTIMIZER.__name__,
        'criterion': PS_CRITETION.__name__,
        'image_size': IMAGE_SIZE,
        'encoder_name': MODEL_ENCODER_NAME,
        'in_channels': IN_CHANNELS,
        'classes': CLASSES,
        'device': DEVICE,
        'default_encoder_weights': MODEL_ENCODER_WEIGHTS,
        'eval_frequency': EVAL_FREQUENCY,
        'visualization_samples': VISUALIZATION_SAMPLES,
    }
}
