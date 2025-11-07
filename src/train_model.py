import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import config
from src.data_loader import prepare_data, get_augmentation

def build_cnn_model():
    """Build CNN model"""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 4
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Classifier
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(len(config.CLASSES), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def plot_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")

def train():
    """Train the model"""
    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    print("\nBuilding model...")
    model = build_cnn_model()
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(config.MODEL_DIR, 'best_model.h5'),
            save_best_only=True
        )
    ]

    # Data augmentation
    datagen = get_augmentation()
    datagen.fit(X_train)

    print("\nTraining model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=config.CLASSES))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))

    # Plot history
    plot_history(history)

    print(f"\nModel saved to {os.path.join(config.MODEL_DIR, 'best_model.h5')}")

if __name__ == '__main__':
    train()
