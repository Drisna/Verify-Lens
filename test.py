import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# Load Extracted Test Features
print("📂 Loading Extracted Test Features...")
with h5py.File('/content/drive/MyDrive/test_features_xception.h5', 'r') as f:
    test_features = np.array(f['features'])
    test_labels = np.array(f['labels'])

print(f"✅ Loaded Test Features: {test_features.shape}")

# Load Trained Model
print("📥 Loading Trained Model...")
model = load_model('/content/drive/MyDrive/deepfake_detector_xception.h5')
print("✅ Model Loaded Successfully")

# Predict Test Data
print("🔍 Predicting Test Data...")
predictions = model.predict(test_features)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Calculate Accuracy
real_indices = np.where(test_labels == 1)[0]
fake_indices = np.where(test_labels == 0)[0]

real_accuracy = accuracy_score(test_labels[real_indices], predicted_labels[real_indices])
fake_accuracy = accuracy_score(test_labels[fake_indices], predicted_labels[fake_indices])
overall_accuracy = accuracy_score(test_labels, predicted_labels)

# Print Accuracy Results
print("\n🎯 TEST RESULTS:")
print("--------------------------------------------------")
print(f"📊 Overall Accuracy: {round(overall_accuracy * 100, 2)}%")
print(f"✅ Real Video Accuracy: {round(real_accuracy * 100, 2)}%")
print(f"❌ Fake Video Accuracy: {round(fake_accuracy * 100, 2)}%")
print("--------------------------------------------------")

# Print Classification Report
print("\n📊 Classification Report:\n", classification_report(test_labels, predicted_labels, target_names=['Fake', 'Real']))

print("✅ Testing Completed.")