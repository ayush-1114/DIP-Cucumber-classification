import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Dynamic Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"  # CHANGED: Point to raw data
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model():
    # 2. Load the Test Dataset
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # CHANGED: Point to the raw test folder
    test_dataset = datasets.ImageFolder(root=RAW_DIR / "test", transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes
    
    filenames = [Path(s[0]).name for s in test_dataset.samples]

    print(f"Testing on {len(test_dataset)} RAW images.")
    print(f"Classes: {class_names}")

    # 3. Rebuild the Model and Load the RAW Trained Weights
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    # CHANGED: Load the mobilenetv2_raw.pth model
    model.load_state_dict(torch.load(MODELS_DIR / "mobilenetv2_raw.pth", map_location=torch.device('cpu')))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 4. Run Inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Generate Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n--- Classification Report (Raw Test Set) ---")
    print(report)

    # CHANGED: Save report to a different text file
    with open(RESULTS_DIR / "dl_classification_report_raw.txt", "w") as f:
        f.write(report)

    # --- 2. Save Per-Image Predictions to CSV ---
    true_labels_str = [class_names[i] for i in all_labels]
    pred_labels_str = [class_names[i] for i in all_preds]
    
    df = pd.DataFrame({
        "filename": filenames,
        "true_label": true_labels_str,
        "predicted_label": pred_labels_str
    })
    
    # CHANGED: Save to dl_predictions_raw.csv
    csv_path = RESULTS_DIR / "dl_predictions_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Per-image predictions saved to {csv_path}")

    # 6. Generate and Plot the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('MobileNetV2 Confusion Matrix (Raw Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # CHANGED: Save plot to a different image file
    plot_path = ROOT_DIR / "results" / "figures" / "dl_confusion_matrix_raw.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to {plot_path}")

if __name__ == "__main__":
    evaluate_model()