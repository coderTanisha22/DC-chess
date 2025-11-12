# models/train.py
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "../data/tiles_labeled"  # relative when running from models/
IMG_SIZE = (128,128)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = None  # auto

def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1],3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    # freeze base
    for layer in base.layers:
        layer.trainable = False
    return model

def main():
    train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                rotation_range=10, width_shift_range=0.05, height_shift_range=0.05,
                                brightness_range=(0.7,1.3), shear_range=0.02, zoom_range=0.08)
    train_flow = train_gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                            class_mode='categorical', subset='training', shuffle=True)
    val_flow = train_gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                            class_mode='categorical', subset='validation', shuffle=False)
    num_classes = train_flow.num_classes
    model = build_model(num_classes)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    # compute class weights (by folder counts)
    labels = train_flow.classes
    classes = np.unique(labels)
    cw = compute_class_weight('balanced', classes=classes, y=labels)
    class_weight = {i: cw[i] for i in range(len(cw))}
    print("Using class_weight:", class_weight)
    # callbacks
    os.makedirs("../models/trained_models", exist_ok=True)
    chk = ModelCheckpoint("../models/trained_models/piece_classifier.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    history = model.fit(train_flow, validation_data=val_flow, epochs=EPOCHS, class_weight=class_weight, callbacks=[chk, es, rl])
    # optional: unfreeze some layers and fine-tune
    base = model.layers[0]
    for layer in base.layers[-30:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history2 = model.fit(train_flow, validation_data=val_flow, epochs=10, class_weight=class_weight, callbacks=[chk, es, rl])
    print("Training complete. Model saved at ../models/trained_models/piece_classifier.h5")

if __name__ == "__main__":
    main()
