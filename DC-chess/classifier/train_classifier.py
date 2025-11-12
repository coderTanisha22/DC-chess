"""
Train a simple MobileNetV2 classifier over data/tiles_labeled directory.
Expect subfolders per class inside data/tiles_labeled.
"""
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

DATA_DIR = "data/tiles_labeled"
IMG_SIZE = (128,128)
BATCH_SIZE = 8
EPOCHS = 10
OUT = "models/classifier/piece_classifier.h5"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1],3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(base.input, out)
    for layer in base.layers:
        layer.trainable = False
    return model

def main():
    train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                   rotation_range=10, width_shift_range=0.05, height_shift_range=0.05,
                                   brightness_range=(0.7,1.3), zoom_range=0.08)
    train_flow = train_gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset='training', class_mode='categorical')
    val_flow = train_gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset='validation', class_mode='categorical')
    num_classes = train_flow.num_classes
    model = build_model(num_classes)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_flow, validation_data=val_flow, epochs=EPOCHS)
    model.save(OUT)
    np.save("models/classifier/class_names.npy", np.array(list(train_flow.class_indices.keys())))
    print("Saved model to", OUT)

if __name__ == "__main__":
    main()
