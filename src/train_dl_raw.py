import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import time
import copy

# 1. Dynamic Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
# CHANGE 1: Point to RAW_DIR instead of PROCESSED_DIR
RAW_DIR = ROOT_DIR / "data" / "raw" 
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Data Augmentation for Training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # PyTorch automatically resizes the huge raw images
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/Test Transforms
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_model():
    # CHANGE 2: Load Datasets from RAW_DIR
    train_dataset = datasets.ImageFolder(root=RAW_DIR / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=RAW_DIR / "val", transform=val_test_transforms)
    
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 4. Handle Class Imbalance
    # CHANGE 3: Count classes in RAW_DIR
    class_counts = [len(list((RAW_DIR / "train" / cls).glob("*.jpg"))) for cls in class_names]
    print(f"Class Counts (Train): {class_counts}")
    
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 5. Load Pre-trained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on: {device}")

    # 6. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 7. Training Loop
    epochs = 10
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 20)
        
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        start_time = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Time: {time.time()-start_time:.1f}s")

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("⭐ New best model saved!")

    # CHANGE 4: Save as mobilenetv2_raw.pth so we don't overwrite the processed model!
    save_path = MODELS_DIR / "mobilenetv2_raw.pth"
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Training Complete. Best Val Acc: {best_val_acc:.2f}%")
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    train_model()