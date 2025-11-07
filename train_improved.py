"""Improved Training - Enhanced techniques for higher accuracy"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import config
from src.data_loader import prepare_data

def get_strong_augmentation():
    """Stronger data augmentation"""
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

def build_improved_model():
    """Build improved model with EfficientNetB3"""
    print("Loading EfficientNetB3 (larger than B0)...")

    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )

    # Freeze base initially
    for layer in base_model.layers:
        layer.trainable = False

    # Improved classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(config.CLASSES), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def train():
    """Improved training strategy"""
    print("=" * 70)
    print("IMPROVED Training - Target: >75% Accuracy")
    print("Using EfficientNetB3 + Strong Augmentation + Multi-phase Training")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    # Build model
    print("\nBuilding improved model...")
    model, base_model = build_improved_model()
    print(f"Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

    # Strong augmentation
    datagen = get_strong_augmentation()
    datagen.fit(X_train)

    # Phase 1: Train head with strong augmentation (25 epochs)
    print("\n" + "=" * 70)
    print("Phase 1: Training head with strong augmentation (25 epochs)")
    print("=" * 70)

    callbacks_phase1 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            os.path.join(config.MODEL_DIR, 'improved_phase1.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=25,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # Phase 2: Unfreeze more layers and train longer (50 epochs)
    print("\n" + "=" * 70)
    print("Phase 2: Fine-tuning (unfreezing last 100 layers, 50 epochs)")
    print("=" * 70)

    # Unfreeze last 100 layers
    for layer in base_model.layers[-100:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

    callbacks_phase2 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            os.path.join(config.MODEL_DIR, 'improved_phase2.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=1
        )
    ]

    history2 = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=callbacks_phase2,
        verbose=1
    )

    # Phase 3: Fine-tune with very low learning rate (25 epochs)
    print("\n" + "=" * 70)
    print("Phase 3: Final fine-tuning (very low LR, 25 epochs)")
    print("=" * 70)

    # Unfreeze all layers
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

    callbacks_phase3 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            os.path.join(config.MODEL_DIR, 'improved_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-9,
            verbose=1
        )
    ]

    history3 = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=25,
        callbacks=callbacks_phase3,
        verbose=1
    )

    # Combine histories
    history = type('obj', (object,), {
        'history': {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'] + history3.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'] + history3.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'] + history3.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'] + history3.history['val_loss']
        }
    })()

    # Evaluate
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{'*' * 70}")
    print(f"{'*' * 70}")
    print(f"***  TEST ACCURACY: {test_acc*100:.2f}%  ***")
    print(f"***  TEST LOSS: {test_loss:.4f}      ***")
    print(f"{'*' * 70}")
    print(f"{'*' * 70}")

    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n" + "=" * 70)
    print("Classification Report:")
    print("=" * 70)
    print(classification_report(y_true, y_pred_classes, target_names=config.CLASSES))

    print("\n" + "=" * 70)
    print("Per-Class Accuracy:")
    print("=" * 70)
    cm = confusion_matrix(y_true, y_pred_classes)
    for i, class_name in enumerate(config.CLASSES):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum() * 100
            print(f"  {class_name:35s}: {class_acc:6.2f}%")

    # Plot
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title(f'Model Accuracy - Final: {test_acc*100:.2f}%', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=150)

    model.save(os.path.join(config.MODEL_DIR, 'improved_final.h5'))

    print("\n" + "=" * 70)
    print("Models saved:")
    print(f"  - Best: improved_best.h5")
    print(f"  - Final: improved_final.h5")
    print("=" * 70)

    if test_acc >= 0.75:
        print("\n" + "=" * 70)
        print("*** SUCCESS! ACHIEVED >75% ACCURACY! ***")
        print("=" * 70)
    elif test_acc >= 0.60:
        print(f"\n*** Good progress! Got {test_acc*100:.2f}% accuracy ***")
    else:
        print(f"\n*** Got {test_acc*100:.2f}% accuracy ***")

if __name__ == '__main__':
    train()
