import os
import random
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "train"
BACKUP_DIR = ROOT_DIR / "data" / "raw" / "train_backup"

MAX_IMAGES = 300
SEED = 42  # For reproducibility

def downsample_class(class_dir):
    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    
    if len(images) <= MAX_IMAGES:
        print(f"✅ {class_dir.name}: {len(images)} images. No downsampling needed.")
        return
        
    print(f"✂️ {class_dir.name}: {len(images)} images found. Downsampling to {MAX_IMAGES}...")
    
    # Shuffle for random selection
    random.seed(SEED)
    random.shuffle(images)
    
    # Create backup folder for this class
    backup_class_dir = BACKUP_DIR / class_dir.name
    backup_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Move the excess images to backup
    images_to_remove = images[MAX_IMAGES:]
    for img_path in images_to_remove:
        shutil.move(str(img_path), str(backup_class_dir / img_path.name))
        
    print(f"   -> Kept {MAX_IMAGES} images. Moved {len(images_to_remove)} to backup.")

if __name__ == "__main__":
    print("Starting downsampling...")
    classes = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    
    for cls_dir in classes:
        downsample_class(cls_dir)
        
    print("\n🎉 Downsampling complete! Your dataset is now balanced.")