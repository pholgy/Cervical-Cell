"""Optimized ResNet50 Training - Faster convergence to >75% accuracy"""
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

def build_optimized_model():
    """Build optimized ResNet50 model"""
    print("Loading ResNet50 with optimizations...")

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )

    # Freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Optimized classification head
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

    # Compile with higher initial learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def train():
    """Optimized training"""
    print("=" * 70)
    print("OPTIMIZED ResNet50 Training - Target: >75% Accuracy")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    # Build model
    print("\nBuilding optimized model...")
    model, base_model = build_optimized_model()
    print(f"Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

    # Optimized callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            os.path.join(config.MODEL_DIR, 'resnet50_optimized.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Stronger augmentation
    datagen = get_augmentation()
    datagen.fit(X_train)

    # Phase 1: Train head with higher LR
    print("\n" + "=" * 70)
    print("Phase 1: Training classification head (15 epochs)")
    print("=" * 70)

    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Unfreeze more layers for better accuracy
    print("\n" + "=" * 70)
    print("Phase 2: Fine-tuning (unfreezing last 60 layers)")
    print("=" * 70)

    # Unfreeze last 60 layers
    for layer in base_model.layers[-60:]:
        layer.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00005),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

    history2 = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=25,
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
    print(f"{'*' * 70}")
    print(f"***  TEST ACCURACY: {test_acc*100:.2f}%  ***")
    print(f"***  TEST LOSS: {test_loss:.4f}      ***")
    print(f"{'*' * 70}")
    print(f"{'*' * 70}")

    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    print("\n" + "=" * 70)
    print("Classification Report:")
    print("=" * 70)
    print(classification_report(y_true, y_pred_classes, target_names=config.CLASSES))

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
    plt.savefig('training_history_optimized.png', dpi=150)
    print(f"\nTraining history saved!")

    # Save final model
    model.save(os.path.join(config.MODEL_DIR, 'resnet50_final.h5'))

    print("\n" + "=" * 70)
    print("Models saved:")
    print(f"  - Best: resnet50_optimized.h5")
    print(f"  - Final: resnet50_final.h5")
    print("=" * 70)

    if test_acc >= 0.75:
        print("\n" + "=" * 70)
        print("*** SUCCESS! ACHIEVED >75% ACCURACY! ***")
        print("=" * 70)
    else:
        print(f"\nGot {test_acc*100:.2f}% accuracy")

if __name__ == '__main__':
    train()
