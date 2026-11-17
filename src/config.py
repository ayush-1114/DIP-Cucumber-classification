import os
from pathlib import Path

# 1. Project Root Directory (Works on any laptop)
ROOT_DIR = Path(__file__).resolve().parent.parent

# 2. Folder Paths
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DUMMY_DIR = ROOT_DIR / "data" / "dummy_data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = ROOT_DIR / "reports"

# 3. Image Settings & Splits
IMG_SIZE = (224, 224)
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # For SN's stratified split script

# 4. The 5 Classes (Ensure exact spelling)
CLASSES = ["Whitefly", "Red_Mite", "Yellow_Mosaic", "Leaf_Miner", "Healthy"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# 5. Dataset Splits
SPLITS = ["train", "val", "test"]

# 6. Safety Net Function
def ensure_dirs():
    """Creates all required project directories if they don't exist."""
    for split in SPLITS:
        (RAW_DIR / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)
        
    directories_to_create = [
        DUMMY_DIR, 
        MODELS_DIR, 
        RESULTS_DIR / "figures", 
        RESULTS_DIR / "metrics", 
        RESULTS_DIR / "storage_reduction",
        REPORTS_DIR / "dip_report_assets",
        REPORTS_DIR / "ml_report_assets"
    ]
    
    for d in directories_to_create:
        d.mkdir(parents=True, exist_ok=True)

# Create __init__.py so 'src' is recognized as a Python package
init_file = ROOT_DIR / "src" / "__init__.py"
init_file.touch(exist_ok=True)