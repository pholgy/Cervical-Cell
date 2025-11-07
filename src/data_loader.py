import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

def load_dataset():
    """Load images from dataset folders"""
    images = []
    labels = []

    # Get all class folders
    class_folders = [
        'im_Dyskeratotic/im_Dyskeratotic',
        'im_Koilocytotic',
        'im_Metaplastic',
        'im_Parabasal/im_Parabasal',
        'im_Superficial-Intermediate/im_Superficial-Intermediate'
    ]

    for idx, folder in enumerate(class_folders):
        folder_path = os.path.join(config.DATA_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found, skipping...")
            continue

        # Get all BMP files
        bmp_files = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]
        print(f"Loading {len(bmp_files)} images from {config.CLASSES[idx]}...")

        for img_file in bmp_files:
            img_path = os.path.join(folder_path, img_file)

            # Read and resize image
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

            images.append(img)
            labels.append(idx)

    print(f"\nTotal images loaded: {len(images)}")
    return np.array(images), np.array(labels)

def prepare_data():
    """Load and split dataset"""
    # Load images
    X, y = load_dataset()

    # Normalize pixel values
    X = X.astype('float32') / 255.0

    # Convert labels to categorical
    y = to_categorical(y, num_classes=len(config.CLASSES))

    # Split: train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(config.VAL_SPLIT + config.TEST_SPLIT), random_state=42
    )

    val_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_augmentation():
    """Data augmentation generator"""
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    return datagen
