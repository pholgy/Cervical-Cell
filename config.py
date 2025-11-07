import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR  # Data folders are in project root
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Model settings - OPTIMIZED FOR 100 EPOCHS
IMG_SIZE = 224
BATCH_SIZE = 32  # Larger batch for faster training
EPOCHS = 100  # More epochs for better accuracy
LEARNING_RATE = 0.001

# Classes
CLASSES = [
    'Dyskeratotic',
    'Koilocytotic',
    'Metaplastic',
    'Parabasal',
    'Superficial-Intermediate'
]

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
