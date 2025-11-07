"""Quick training test with fewer epochs"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF warnings

import sys
from src.train_model import build_cnn_model, plot_history
from src.data_loader import prepare_data, get_augmentation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import config

print("=" * 60)
print("Quick Training Test (10 epochs)")
print("=" * 60)

# Reduce epochs for quick test
config.EPOCHS = 10

print("\nLoading and preparing data...")
(X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

print("\nBuilding model...")
model = build_cnn_model()

print("\nTraining...")
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(
        os.path.join(config.MODEL_DIR, 'best_model.h5'),
        save_best_only=True,
        verbose=1
    )
]

datagen = get_augmentation()
datagen.fit(X_train)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=config.EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\nEvaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

plot_history(history)
print("\nModel saved! Check 'training_history.png' for results.")
