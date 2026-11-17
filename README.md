# Cucumber Disease Classification

## Team Roles
*   **AP**: DIP Pipeline & Deep Learning Architect
*   **SN**: Data Engineer & Classical ML Specialist
*   **PJ**: Model Analysis & Reporting Engineer

## Folder Structure
*   `data/`: Datasets (Raw and Processed)
*   `src/`: Python scripts (DIP, ML, DL, Evaluation)
*   `models/`: Saved model weights
*   `results/`: Metrics, CSV files, and plots

## Environment Setup
1. Create virtual environment: `python -m venv mlee-env`
2. Activate it: `.\mlee-env\Scripts\Activate.ps1`
3. Install dependencies: `pip install -r requirements.txt`
<!-- 
external_data/: This holds the heavy, raw datasets (the Roboflow YOLO files, the tea dataset, the mosaic dataset). We keep this out of GitHub so it doesn't break the repository.

src/config.py: The "single source of truth." It holds the paths, image sizes (224x224), and class names so all three of you (AP, SN, PJ) are on the same page.

src/crop_pests.py (The New File): Purpose: The Roboflow dataset gives us full images of leaves with tiny bounding boxes drawn around the bugs. Our DIP pipeline needs only the bug/lesion close-up to learn from. This script reads the YOLO .txt files, cuts out (crops) the Whitefly and Leaf Miner bugs, and saves them as individual .jpg files into data/raw/train/Whitefly/ and data/raw/train/Leaf_Miner/.

src/crop_yolo.py (The Old File): Purpose: This was an earlier draft. It is obsolete because we now have the exact class IDs from your data.yaml. You can safely delete this file. -->