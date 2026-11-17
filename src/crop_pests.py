import cv2
import os
import glob
from pathlib import Path

# --- CONFIGURATION ---
# Point this to the dataset folder you extracted
RAW_DATASET_DIR = Path("external_data/roboflow_raw_data/dataset")

# Where to save the cropped images
BASE_OUT_DIR = Path("data/raw/train")

# The Class IDs from your data.yaml
CLASS_MAPPING = {
    3: "Leaf_Miner",  # Serpentine leafminer
    4: "Whitefly"     # Whitefly
}

def crop_yolo_boxes(split="train"):
    images_dir = RAW_DATASET_DIR / split / "images"
    labels_dir = RAW_DATASET_DIR / split / "labels"

    # Find all images in the split
    image_paths = glob.glob(str(images_dir / "*.jpg")) + glob.glob(str(images_dir / "*.png"))
    
    print(f"Found {len(image_paths)} images in {split}. Processing...")
    cropped_count = 0

    for img_path in image_paths:
        img_name = Path(img_path).stem
        label_path = labels_dir / f"{img_name}.txt"

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape

        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            class_id = int(parts[0])
            if class_id not in CLASS_MAPPING: continue

            # YOLO format: x_center, y_center, width, height (normalized 0-1)
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            # Convert to absolute pixel coordinates
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop with slight padding
            pad = 15
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w, x2+pad), min(h, y2+pad)

            cropped_img = img[y1:y2, x1:x2]

            if cropped_img.size == 0: continue

            # Save to data/raw/train/ClassName/
            class_name = CLASS_MAPPING[class_id]
            out_path = BASE_OUT_DIR / class_name / f"{split}_{img_name}_crop{i}.jpg"
            cv2.imwrite(str(out_path), cropped_img)
            cropped_count += 1

    print(f"✅ Success! Saved {cropped_count} cropped {split} images to {BASE_OUT_DIR}")

if __name__ == "__main__":
    # Process train, valid, and test splits
    for split in ["train", "valid", "test"]:
        crop_yolo_boxes(split)