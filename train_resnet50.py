"""Train with ResNet50 Transfer Learning"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import config
from src.data_loader import prepare_data, get_augmentation

def build_resnet50_model():
    """Build ResNet50 transfer learning model"""
    print("Loading pre-trained ResNet50...")

    # Load ResNet50 with ImageNet weights
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )

    # Freeze all base layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(config.CLASSES), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def plot_history(history, filename='training_history_resnet50.png'):
    """Plot training history"""
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('ResNet50 - Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('ResNet50 - Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nTraining history saved to {filename}")

def train():
    """Train ResNet50 model"""
    print("=" * 70)
    print("ResNet50 Transfer Learning Training")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    # Build model
    print("\nBuilding ResNet50 model...")
    model, base_model = build_resnet50_model()
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(config.MODEL_DIR, 'resnet50_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Data augmentation
    datagen = get_augmentation()
    datagen.fit(X_train)

    # Phase 1: Train top layers
    print("\n" + "=" * 70)
    print("Phase 1: Training classification head (20 epochs)")
    print("=" * 70)

    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tuning
    print("\n" + "=" * 70)
    print("Phase 2: Fine-tuning (unfreezing last 50 layers)")
    print("=" * 70)

    # Unfreeze last 50 layers
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Trainable parameters after unfreezing: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

    history2 = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    # Combine histories
    history = type('obj', (object,), {
        'history': {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
    })()

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n*** Test Accuracy: {test_acc*100:.2f}% ***")
    print(f"    Test Loss: {test_loss:.4f}")

    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    print("\n" + "=" * 70)
    print("Classification Report:")
    print("=" * 70)
    print(classification_report(y_true, y_pred_classes, target_names=config.CLASSES))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))

    # Per-class accuracy
    print("\n" + "=" * 70)
    print("Per-Class Accuracy:")
    print("=" * 70)
    cm = confusion_matrix(y_true, y_pred_classes)
    for i, class_name in enumerate(config.CLASSES):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum() * 100
            print(f"  {class_name:35s}: {class_acc:6.2f}%")

    # Plot history
    plot_history(history)

    # Save final model
    model.save(os.path.join(config.MODEL_DIR, 'resnet50_final.h5'))

    print("\n" + "=" * 70)
    print(f"Best model saved: {os.path.join(config.MODEL_DIR, 'resnet50_best.h5')}")
    print(f"Final model saved: {os.path.join(config.MODEL_DIR, 'resnet50_final.h5')}")
    print("=" * 70)

    if test_acc >= 0.75:
        print("\n*** SUCCESS! Achieved >75% accuracy! ***")
    else:
        print(f"\n*** Almost there! Got {test_acc*100:.2f}% accuracy ***")

if __name__ == '__main__':
    train()
