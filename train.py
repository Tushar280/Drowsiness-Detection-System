import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


DATASET_DIR = "data/processed_dataset" 
IMG_SIZE = 64

def build_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax') # 3 classes: Alert, Drowsy_Eyes_Closed, Drowsy_Yawning
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():

    
    if not os.path.exists(os.path.join(DATASET_DIR, 'train')):
        print(f"Training directory not found at {os.path.join(DATASET_DIR, 'train')}")
        print("Please prepare your dataset first (Crop faces and split into train/val folders).")
        return

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )

    model = build_model()
    model.summary()

    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator
    )

    model.save('drowsiness_detection_model.keras')
    print("Model saved to drowsiness_detection_model.keras")

    # Predict on validation set
    print("\n--- Model Evaluation ---")
    val_loss, val_acc = model.evaluate(val_generator)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")

    Y_pred = model.predict(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    print('\nConfusion Matrix')
    print(confusion_matrix(val_generator.classes, y_pred))
    print('\nClassification Report')
    target_names = list(val_generator.class_indices.keys())
    print(classification_report(val_generator.classes, y_pred, target_names=target_names))

    # Plot Accuracy and Loss
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    print("Graphs saved to 'evaluation_metrics.png'")

if __name__ == "__main__":
    train_model()
