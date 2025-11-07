"""Simple test script to verify data loading works"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_loader import load_dataset
import config

print("=" * 50)
print("Testing Data Loading")
print("=" * 50)

print(f"\nExpected classes: {config.CLASSES}")
print(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
print(f"Data directory: {config.DATA_DIR}")

print("\nAttempting to load dataset...")
try:
    X, y = load_dataset()
    print(f"\n✓ Success!")
    print(f"Loaded {len(X)} images")
    print(f"Image shape: {X[0].shape}")
    print(f"Labels shape: {y.shape}")

    # Count per class
    print("\nImages per class:")
    for i, class_name in enumerate(config.CLASSES):
        count = sum(y == i)
        print(f"  {class_name}: {count}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
