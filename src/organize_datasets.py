import os
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT_DIR / "external_data"
DEST_BASE_DIR = ROOT_DIR / "data" / "raw" / "train"

IMG_EXTS = {".jpg", ".jpeg", ".png"}
MAX_IMAGES_PER_CLASS = 1500 # Prevents class imbalance and insane training times

def find_folder_by_name(root_path, folder_name):
    for path in root_path.rglob(folder_name):
        if path.is_dir():
            return path
    return None

def copy_images(class_name, source_dir):
    dest_dir = DEST_BASE_DIR / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if source_dir is None or not source_dir.exists():
        print(f"❌ Error: Could not find the folder for {class_name}")
        return 0
        
    count = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix.lower() in IMG_EXTS:
                
                # Stop if we hit the limit
                if count >= MAX_IMAGES_PER_CLASS:
                    break
                    
                src_file = os.path.abspath(Path(root) / file)
                dest_file = os.path.abspath(dest_dir / f"{class_name.lower()}_{count:04d}{Path(file).suffix.lower()}")
                
                # THE MAGIC FIX: Bypass Windows 260-character limit
                if os.name == 'nt':
                    src_file = "\\\\?\\" + src_file
                    dest_file = "\\\\?\\" + dest_file

                try:
                    # Read and write manually to bypass the OS path error
                    with open(src_file, 'rb') as f_src, open(dest_file, 'wb') as f_dst:
                        shutil.copyfileobj(f_src, f_dst)
                    count += 1
                except Exception as e:
                    print(f"⚠️ Skipping {file}: {e}")
                    
        if count >= MAX_IMAGES_PER_CLASS:
            break
                
    print(f"✅ Copied {count} images to data/raw/train/{class_name}")
    return count

if __name__ == "__main__":
    print("Searching for datasets in external_data...")
    
    healthy_src = find_folder_by_name(EXTERNAL_DIR, "Healthy Leaves")
    mosaic_src = find_folder_by_name(EXTERNAL_DIR, "Cucumber Mosaic Virus")
    red_mite_src = find_folder_by_name(EXTERNAL_DIR, "03-Red Spider-Tea-Disease")
    
    PATHS_TO_COPY = {
        "Healthy": healthy_src,
        "Yellow_Mosaic": mosaic_src,
        "Red_Mite": red_mite_src
    }
    
    total = 0
    for class_name, source_path in PATHS_TO_COPY.items():
        print(f"\nProcessing {class_name}...")
        total += copy_images(class_name, source_path)
        
    print(f"\n🎉 Done! Copied a total of {total} images.")