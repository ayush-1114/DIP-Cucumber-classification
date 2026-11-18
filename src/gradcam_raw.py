import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
from pathlib import Path
import random

# 1. Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
TEST_DIR = ROOT_DIR / "data" / "raw" / "test"
OUT_DIR = ROOT_DIR / "results" / "figures" / "gradcam_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Rebuild the Model Architecture
def load_model():
    model = models.mobilenet_v2(weights=None) 
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 5) 
    
    model.load_state_dict(torch.load(MODELS_DIR / "mobilenetv2_raw.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# 3. Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam():
    model = load_model()
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    classes = ['Healthy', 'Leaf_Miner', 'Red_Mite', 'Whitefly', 'Yellow_Mosaic']
    
    print("Generating Grad-CAM heatmaps for the RAW model...")
    
    for cls in classes:
        cls_dir = TEST_DIR / cls
        images = list(cls_dir.glob("*.jpg"))
        if not images: continue
        
        img_path = random.choice(images)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Prepare tensor for the model
        input_tensor = transform(img_rgb).unsqueeze(0)
        
        # Generate Heatmap (this will be 224x224)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # THE FIX: Resize the original raw image to 224x224 to match the heatmap
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Overlay heatmap on the resized image
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        # Save output
        out_path = OUT_DIR / f"{cls}_gradcam_raw.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved Grad-CAM for {cls} at {out_path}")

if __name__ == "__main__":
    generate_gradcam()