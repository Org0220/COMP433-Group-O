import os
import torch

# Base directory
BASE_DIR = os.getcwd()
# Data directories
DATA_DIR = f'{BASE_DIR}\\data'
IMAGES_DIR = f'{BASE_DIR}\\data\\Sliced_Images'  # Update as needed 
LABELS_FILE = os.path.join(DATA_DIR, 'image_class_mapping.csv')

# Splits directory
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')

# Training parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# DataLoader parameters
BATCH_SIZE_BYOL = 64
BATCH_SIZE_SUPERVISED = 16

# Training settings
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
LEARNING_RATE_UNFROZEN = 1e-4
LEARNING_RATE_FROZEN = 1e-3  # Higher learning rate when encoder is frozen
SAVE_PATH = os.path.join(DATA_DIR, 'best_byol_model.pth')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
