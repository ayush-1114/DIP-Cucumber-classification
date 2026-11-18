import os
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_PATH = ROOT_DIR / "models" / "mobilenetv2_cucumber.pth"
METRICS_DIR = ROOT_DIR / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024) # Convert to MB

if __name__ == "__main__":
    raw_size = get_folder_size(RAW_DIR)
    proc_size = get_folder_size(PROCESSED_DIR)
    
    reduction = 0.0
    if raw_size > 0:
        reduction = ((raw_size - proc_size) / raw_size) * 100
        
    model_size = 0.0
    if MODEL_PATH.exists():
        model_size = MODEL_PATH.stat().st_size / (1024 * 1024)

    print("--- Storage Footprint & Compression Metrics ---")
    print(f"Raw Dataset Size:       {raw_size:.2f} MB")
    print(f"Processed Dataset Size: {proc_size:.2f} MB")
    print(f"Storage Reduction:      {reduction:.2f}%")
    print(f"MobileNetV2 Model Size: {model_size:.2f} MB")

    # Save to CSV for the report
    df = pd.DataFrame({
        "Metric": ["Raw Data Size (MB)", "Processed Data Size (MB)", "Storage Reduction (%)", "Model Size (MB)"],
        "Value": [f"{raw_size:.2f}", f"{proc_size:.2f}", f"{reduction:.2f}%", f"{model_size:.2f}"]
    })
    
    csv_path = METRICS_DIR / "storage_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved to {csv_path}")