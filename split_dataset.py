import os
import shutil
import random

# ✅ Define Paths
DATASET_PATH = r"D:\drisna\.vscode\verifylens\dataset_processed"
SPLIT_PATH = r"D:\drisna\.vscode\verifylens\split_dataset"

# ✅ Define Split Ratios
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ✅ Categories (Real & Fake)
CATEGORIES = ["real", "fake"]

# ✅ Ensure split directories exist
for split in ["train", "valid", "test"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(SPLIT_PATH, split, category), exist_ok=True)

# ✅ Function to split dataset
def split_and_copy(category):
    category_path = os.path.join(DATASET_PATH, category)

    if not os.path.exists(category_path):
        print(f" Warning: {category_path} does not exist. Skipping...")
        return

    # ✅ Get all video folders
    videos = sorted(os.listdir(category_path))
    random.shuffle(videos)  # Shuffle for randomness

    total_videos = len(videos)
    train_count = int(total_videos * TRAIN_RATIO)
    valid_count = int(total_videos * VALID_RATIO)

    split_map = {
        "train": videos[:train_count],
        "valid": videos[train_count:train_count + valid_count],
        "test": videos[train_count + valid_count:]
    }

    for split, video_list in split_map.items():
        for video_folder in video_list:
            src_folder = os.path.join(category_path, video_folder)
            dst_folder = os.path.join(SPLIT_PATH, split, category, video_folder)

            if os.path.exists(dst_folder):
                print(f"Skipping {dst_folder}, already exists.")
                continue

            shutil.copytree(src_folder, dst_folder)  # ✅ Copy entire folder
            print(f" Copied {video_folder} to {split}/{category}")

# ✅ Run splitting for both categories
for cat in CATEGORIES:
    split_and_copy(cat)

print("\n Dataset successfully split into Train, Validation, and Test sets!")
