import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import gdown

MODEL_ID = "1gzAAR5FLhrZVFwRIpKYYbEX0B7GcOwG3"
MODEL_PATH = "deepfake_detector_xception.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

download_model()


app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


try:
    model = load_model("deepfake_detector_xception.h5")
    print(" Model loaded successfully.")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None


try:
    base_model = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    print("Xception feature extractor loaded.")
except Exception as e:
    print(f" Error loading feature extractor: {e}")
    feature_extractor = None

def extract_features_from_video(video_path):
    """Extract features from a video using Xception model."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    max_frames = 30  

    if not cap.isOpened():
        print(f" Error opening video: {video_path}")
        return None

    print("\n Extracting frames & features from video...")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype("float32") / 255.0
        frame = img_to_array(frame)
        frame = np.expand_dims(frame, axis=0)

        
        feature = feature_extractor.predict(frame, verbose=0)
        frames.append(feature.flatten())

        frame_count += 1
        print(f" Processed Frame {frame_count} / {max_frames}")

    cap.release()

    if not frames:
        print(" No frames extracted!")
        return None

   
    print(f" Extracted Feature Vector (First 5 Values): {frames[0][:5]}")

    
    video_features = np.mean(frames, axis=0)
    return np.expand_dims(video_features, axis=0)

@app.route("/")
def home():
    """Render the homepage."""
    return render_template("index.html")

@app.route("/upload")
def upload_form():
    """Render the upload page."""
    return render_template("upload.html")

@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Handle video upload and process deepfake detection."""
    if "video" not in request.files:
        print(" No video uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    video = request.files["video"]
    if video.filename == "":
        print(" No file selected")
        return jsonify({"error": "No selected file"}), 400

    
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
    video.save(video_path)

    print(f"\n--- Processing Video: {video.filename} ---")

   
    features = extract_features_from_video(video_path)
    if features is None:
        return jsonify({"error": "Failed to extract features from video"}), 500

    
    if model is None:
        print(" ERROR: Model not loaded! Prediction skipped.")
        return jsonify({"error": "Model not loaded"}), 500

    print("\n Running Deepfake Detection Model...")

    
    prediction = model.predict(features)[0][0]

    print(f" Raw Model Prediction Output: {prediction:.4f}")  

   
    if prediction > 0.5:
        result = "Real"
        confidence = prediction * 100
    else:
        result = "Fake"
        confidence = (1 - prediction) * 100

    print(f" Model Predicted: {result} with {confidence:.2f}% confidence\n")

    return jsonify({
        "result": result,
        "confidence": f"{confidence:.2f}%",
        "file_url": f"/uploads/{video.filename}" 
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded video file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    port=int(os.environ.get("PORT".5000))
    app.run(host="0.0.0.0",port=port, debug=False)
    
