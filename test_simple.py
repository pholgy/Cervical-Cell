"""Simple test without TensorFlow"""
import os
import cv2
import config

print("=" * 50)
print("Simple Data Loading Test")
print("=" * 50)

class_folders = [
    'im_Dyskeratotic/im_Dyskeratotic',
    'im_Koilocytotic',
    'im_Metaplastic',
    'im_Parabasal/im_Parabasal',
    'im_Superficial-Intermediate/im_Superficial-Intermediate'
]

total_images = 0
for idx, folder in enumerate(class_folders):
    folder_path = os.path.join(config.DATA_DIR, folder)

    if os.path.exists(folder_path):
        bmp_files = [f for f in os.listdir(folder_path) if f.endswith('.bmp')]
        print(f"{config.CLASSES[idx]}: {len(bmp_files)} images")
        total_images += len(bmp_files)
    else:
        print(f"{config.CLASSES[idx]}: FOLDER NOT FOUND")

print(f"\nTotal: {total_images} images")
print("=" * 50)
print("\nNote: Install TensorFlow for full training:")
print("  You need Python 3.11 or 3.12 (you have 3.14)")
print("  Then: pip install tensorflow")
