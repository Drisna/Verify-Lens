# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import h5py

dataset_path = '/content/drive/MyDrive/dataset_splits'

base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)


def extract_and_save_features_hdf5(directory, output_file):
    feature_list = []
    label_list = []
    total_videos = sum([len(os.listdir(os.path.join(directory, label))) for label in ['real', 'fake'] if os.path.exists(os.path.join(directory, label))])
    processed_videos = 0

    for label in ['real', 'fake']:
        folder_path = os.path.join(directory, label)
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        for video_folder in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_folder)
            frame_features = []

            if not os.path.isdir(video_path):
                continue

            frame_limit = 30 if label == 'real' else 10  

            for frame in sorted(os.listdir(video_path))[:frame_limit]:
                img_path = os.path.join(video_path, frame)
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img) / 255.0  
                img_array = np.expand_dims(img_array, axis=0)

                feature = feature_extractor.predict(img_array, verbose=0)
                frame_features.append(feature.flatten())

            if frame_features:
                video_feature = np.mean(frame_features, axis=0)  
                feature_list.append(video_feature)
                label_list.append(1 if label == 'real' else 0)

            processed_videos += 1
            print(f"✅ Processed {processed_videos}/{total_videos} videos")

    feature_array = np.array(feature_list)
    label_array = np.array(label_list)

   
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('features', data=feature_array)
        f.create_dataset('labels', data=label_array)

    print(f"🎉 Features saved to {output_file}")


extract_and_save_features_hdf5(f'{dataset_path}/train', '/content/drive/MyDrive/train_features_xception.h5')
extract_and_save_features_hdf5(f'{dataset_path}/val', '/content/drive/MyDrive/val_features_xception.h5')

print("✅ Feature extraction completed.")
