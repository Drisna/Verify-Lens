import os
import cv2
import numpy as np

# Define dataset and output paths
DATASET_PATH = r"D:\verifylens\kagglehub"
OUTPUT_PATH = r"D:\verifylens\dataset_processed"

# Define real and fake video directories
CATEGORIES = {
    "real": os.path.join(DATASET_PATH, "real"),
    "fake": os.path.join(DATASET_PATH, "fake")
}

# Create output folders
for category in CATEGORIES.keys():
    os.makedirs(os.path.join(OUTPUT_PATH, category), exist_ok=True)

# Frame extraction intervals to balance dataset
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
    frame_count = 0
    current_time = 0.0
    extracted_frames = []

    print(f" Extracting frames from {filename} ({category})")
    video_frames_folder = os.path.join(OUTPUT_PATH, category, filename.split('.')[0])
    os.makedirs(video_frames_folder, exist_ok=True)

    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype("float32") / 255.0
        frame = (frame * 255).astype(np.uint8)  # Convert back to 8-bit for saving

        # Save frame
        frame_filename = os.path.join(video_frames_folder, f"{filename.split('.')[0]}_frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        extracted_frames.append(frame_filename)
        
        frame_count += 1
        current_time += frame_time_interval

    cap.release()
    return extracted_frames

def process_dataset():
    for category, category_path in CATEGORIES.items():
        if not os.path.exists(category_path):
            print(f"Skipping {category}: Folder not found.")
            continue

        for video_file in os.listdir(category_path):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(category_path, video_file)
                print(f"Processing: {video_file}")
                extracted_frames = extract_frames(video_path, video_file, category)
                
                if extracted_frames:
                    print(f" Extracted {len(extracted_frames)} frames from {video_file}")
                else:
                    print(f" Skipped {video_file}: No frames extracted")

if __name__ == "__main__":
    print("Starting dataset frame extraction and preprocessing...")
    process_dataset()
    print("Dataset processing complete!")
