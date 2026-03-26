"""
Train a MobileNetV2 classifier on data/tiles_labeled.
Each subdirectory inside data/tiles_labeled represents a class.
"""

import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# Configuration
# -------------------------
DATA_DIR = "data/tiles_labeled"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 10
MODEL_OUT = "models/classifier/piece_classifier.h5"
CLASS_NAMES_OUT = "models/classifier/class_names.npy"

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)


# -------------------------
# Model Builder
# -------------------------
def build_model(num_classes):
    """Build MobileNetV2 classifier with frozen base."""
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)

    # Freeze base model
    for layer in base.layers:
        layer.trainable = False

    return model


# -------------------------
# Training
# -------------------------
def main():
    # Data generators
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.15,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=(0.7, 1.3),
        zoom_range=0.08,
    )

    train_flow = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="training",
        class_mode="categorical"
    )

    val_flow = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="validation",
        class_mode="categorical"
    )

    num_classes = train_flow.num_classes
    model = build_model(num_classes)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=EPOCHS
    )

    # Save model + class names
    model.save(MODEL_OUT)

    class_names = list(train_flow.class_indices.keys())
    np.save(CLASS_NAMES_OUT, np.array(class_names))

    print(f"Model saved to {MODEL_OUT}")
    print(f"Class names saved to {CLASS_NAMES_OUT}")


if __name__ == "__main__":
    main()
