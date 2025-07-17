import os
import cv2
import numpy as np

# Define dataset and output paths
DATASET_PATH = r"D:\drisna\.vscode\verifylens\kagglehub"
OUTPUT_PATH = r"D:\drisna\.vscode\verifylens\dataset_processed"

# Define real and fake video directories
CATEGORIES = {
    "real": os.path.join(DATASET_PATH, "real"),
    "fake": os.path.join(DATASET_PATH, "fake")
}

# Create output folders
for category in CATEGORIES.keys():
    os.makedirs(os.path.join(OUTPUT_PATH, category), exist_ok=True)

# Frame extraction intervals (adjusted for dataset imbalance)
FRAME_INTERVALS = {
    "real": 0.1,  # Extract every 0.1 seconds (more frames for real)
    "fake": 0.2   # Extract every 0.2 seconds (fewer frames for fake)
}

def extract_frames(video_path, filename, category):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  

    if frame_rate <= 0:
        cap.release()
        return []

    frame_time_interval = FRAME_INTERVALS[category]  
    extracted_frames = []
    frame_index = 0
    current_time = 0.0  

    video_frames_folder = os.path.join(OUTPUT_PATH, category, filename.split('.')[0])
    os.makedirs(video_frames_folder, exist_ok=True)

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)  
        ret, frame = cap.read()

        if not ret:
            break  

        # Save original frame
        frame_filename = os.path.join(video_frames_folder, f"{filename.split('.')[0]}_frame_{frame_index}.jpg")
        cv2.imwrite(frame_filename, frame)
        extracted_frames.append(frame_filename)

        # Apply preprocessing and save processed frame
        processed_frame = preprocess_frame(frame)
        processed_filename = os.path.join(video_frames_folder, f"{filename.split('.')[0]}_frame_{frame_index}_processed.jpg")
        cv2.imwrite(processed_filename, processed_frame)

        frame_index += 1
        current_time += frame_time_interval  

    cap.release()
    return extracted_frames

def preprocess_frame(frame):
    """Preprocessing: Resize and Normalize (No Grayscale)"""
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to 224x224
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return (frame_normalized * 255).astype(np.uint8)  # Convert back to 8-bit for saving

def process_dataset():
    for category, category_path in CATEGORIES.items():
        if not os.path.exists(category_path):
            print(f"Skipping {category}: Folder not found.")
            continue

        for video_file in os.listdir(category_path):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(category_path, video_file)
                print(f"Processing: {video_file}")
                extract_frames(video_path, video_file, category)

if __name__ == "__main__":
    print("Starting dataset frame extraction and preprocessing...")
    process_dataset()
    print("Dataset processing complete!")
