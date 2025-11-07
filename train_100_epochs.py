"""Train ResNet50 for 100 epochs - Maximum accuracy"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import config
from src.data_loader import prepare_data, get_augmentation

def build_model():
    """Build ResNet50 model"""
    print("Loading ResNet50...")

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )

    # Freeze base
    for layer in base_model.layers:
        layer.trainable = False

    # Classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
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
    """Train for 100 epochs"""
    print("=" * 70)
    print("ResNet50 Training - 100 EPOCHS - Target: >80% Accuracy")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    # Build model
    print("\nBuilding model...")
    model, base_model = build_model()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            os.path.join(config.MODEL_DIR, 'resnet50_100epochs.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Augmentation
    datagen = get_augmentation()
    datagen.fit(X_train)

    # Phase 1: Train head (30 epochs)
    print("\n" + "=" * 70)
    print("Phase 1: Training head (30 epochs)")
    print("=" * 70)

    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tune (70 epochs)
    print("\n" + "=" * 70)
    print("Phase 2: Fine-tuning (70 epochs, unfreezing last 70 layers)")
    print("=" * 70)

    # Unfreeze last 70 layers
    for layer in base_model.layers[-70:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=0.00003),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=70,
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

    # Evaluate
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{'*' * 70}")
    print(f"***  TEST ACCURACY: {test_acc*100:.2f}%  ***")
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
    plt.savefig('training_history_100epochs.png', dpi=150)

    model.save(os.path.join(config.MODEL_DIR, 'resnet50_final_100epochs.h5'))

    print("\n" + "=" * 70)
    if test_acc >= 0.80:
        print("*** EXCELLENT! ACHIEVED >80% ACCURACY! ***")
    elif test_acc >= 0.75:
        print("*** SUCCESS! ACHIEVED >75% ACCURACY! ***")
    else:
        print(f"Got {test_acc*100:.2f}% accuracy")
    print("=" * 70)

if __name__ == '__main__':
    train()
