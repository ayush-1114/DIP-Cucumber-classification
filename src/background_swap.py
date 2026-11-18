import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"
MODELS_DIR = ROOT_DIR / "models"

# Standard transforms to load the images
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def evaluate_ood(model_path, data_dir, apply_noise=False):
    # 1. Load Model
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load(MODELS_DIR / model_path, map_location='cpu'))
    model.eval()

    # 2. Load Data
    dataset = datasets.ImageFolder(root=data_dir / "test", transform=test_transforms)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            
            if apply_noise:
                # THE BACKGROUND SWAP: Add extreme random noise
                noise = torch.randn_like(inputs) * 0.8 
                inputs = inputs + noise 
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return (correct / total) * 100

if __name__ == "__main__":
    print("--- 1. Normal Test Set (Clean Images) ---")
    raw_clean = evaluate_ood('mobilenetv2_raw.pth', RAW_DIR, False)
    proc_clean = evaluate_ood('mobilenetv2_cucumber.pth', PROCESSED_DIR, False)
    print(f"Raw Model Accuracy:       {raw_clean:.2f}%")
    print(f"Processed Model Accuracy: {proc_clean:.2f}%")

    print("\n--- 2. Out-Of-Distribution Test (Extreme Background Noise) ---")
    raw_noisy = evaluate_ood('mobilenetv2_raw.pth', RAW_DIR, True)
    proc_noisy = evaluate_ood('mobilenetv2_cucumber.pth', PROCESSED_DIR, True)
    print(f"Raw Model Accuracy:       {raw_noisy:.2f}% (Dropped {raw_clean - raw_noisy:.2f}%)")
    print(f"Processed Model Accuracy: {proc_noisy:.2f}% (Dropped {proc_clean - proc_noisy:.2f}%)")
    
    # NEW: Save to CSV
    results_df = pd.DataFrame({
        "Condition": ["Clean", "Noisy (OOD)"],
        "Raw_Accuracy": [f"{raw_clean:.2f}%", f"{raw_noisy:.2f}%"],
        "Processed_Accuracy": [f"{proc_clean:.2f}%", f"{proc_noisy:.2f}%"],
        "Raw_Drop": ["-", f"{raw_clean - raw_noisy:.2f}%"],
        "Processed_Drop": ["-", f"{proc_clean - proc_noisy:.2f}%"]
    })
    
    csv_path = ROOT_DIR / "results" / "metrics" / "ood_background_swap_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")