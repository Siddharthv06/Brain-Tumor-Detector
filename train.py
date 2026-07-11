# ==========================================
# Brain Tumor Detection - Training Script
# ==========================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ===========================
# Configuration
# ===========================

DATASET_PATH = "dataset"

IMAGE_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 30

CLASSES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

print("=" * 60)
print("Brain Tumor Detection")
print("=" * 60)

print("TensorFlow Version :", tf.__version__)
# ==========================================
# Load Dataset
# ==========================================

images = []
labels = []

print("\nLoading dataset...\n")

for class_index, class_name in enumerate(CLASSES):

    class_path = os.path.join(DATASET_PATH, class_name)

    print(f"Loading {class_name}...")

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image / 255.0

        images.append(image)

        labels.append(class_index)

print("\nDataset Loaded Successfully!")

images = np.array(images, dtype="float32")
labels = np.array(labels)

print("Images Shape :", images.shape)
print("Labels Shape :", labels.shape)

# ==========================================
# Split Dataset
# ==========================================

# First split: 70% Train, 30% Temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    images,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

# Second split:
# 30% becomes 15% Validation + 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Convert labels to one-hot encoding for categorical crossentropy
y_train = to_categorical(y_train, num_classes=len(CLASSES))
y_val = to_categorical(y_val, num_classes=len(CLASSES))
y_test = to_categorical(y_test, num_classes=len(CLASSES))

print("\nDataset Split Complete")
print("-" * 40)

print("Training Images   :", len(X_train))
print("Validation Images :", len(X_val))
print("Testing Images    :", len(X_test))

# ==========================================
# Data Augmentation
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
    tf.keras.layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
    tf.keras.layers.RandomBrightness(factor=0.10)
])

print("\nData Augmentation Ready")

# ==========================================
# Build Custom CNN Model
# ==========================================

model = Sequential([

    # Block 1
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    data_augmentation,

    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.30),

    # Block 4
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.30),

    # Fully Connected Layers
    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.50),

    Dense(256, activation='relu'),
    Dropout(0.40),

    Dense(128, activation='relu'),
    Dropout(0.30),

    Dense(4, activation='softmax')

])

print("\nModel Created Successfully")

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Compiled")

model.summary()

# ==========================================
# Callbacks
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ==========================================
# Start Training
# ==========================================

print("\nStarting Training...\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stop,
        checkpoint
    ]

)

print("\nTraining Completed!")

# ==========================================
# Evaluate Model
# ==========================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# ==========================================
# Predictions
# ==========================================

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.argmax(y_test, axis=1)

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASSES
    )
)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

plt.figure(figsize=(8,8))

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

# ==========================================
# Accuracy Graph
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.legend()

plt.savefig("accuracy.png")

plt.show()

# ==========================================
# Loss Graph
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.legend()

plt.savefig("loss.png")

plt.show()

print("\n========================================")
print("Project Completed Successfully")
print("========================================")
print("Saved Files:")
print("model.keras")
print("accuracy.png")
print("loss.png")
print("confusion_matrix.png")