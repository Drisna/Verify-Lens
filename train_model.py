import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import h5py
import numpy as np
import os

from google.colab import drive
drive.mount('/content/drive')

def load_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        features = f['features'][:]
        labels = f['labels'][:]
    return features, labels


train_features, train_labels = load_hdf5('/content/drive/MyDrive/train_features_xception.h5')
val_features, val_labels = load_hdf5('/content/drive/MyDrive/val_features_xception.h5')

model = Sequential([
    tf.keras.layers.Input(shape=(train_features.shape[1],)),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_features, train_labels, validation_data=(val_features, val_labels), epochs=20, batch_size=32)

model_save_path = "/content/drive/MyDrive/deepfake_detector_xception.h5"
model.save(model_save_path)

print(f"✅ Model training completed and saved at: {model_save_path}")
