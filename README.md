# Cucumber Disease Classification

Start with [the repository guide](REPO_GUIDE.md) for file purposes, current commands, model/result locations and earlier experiments.

See the [DIP submission audit](reports/DIP_PROJECT_STATUS.md), [live demonstration guide](reports/PRESENTATION_GUIDE.md) and [DIP technical report](reports/DIP_TECHNICAL_REPORT.md).

## Current corrected workflow

The corrected internal test has 135 views. Accuracy is 98.52% for raw MobileNetV2, 95.56% for DIP MobileNetV2 and 82.96% for DIP SVM. Validation selected DIP MobileNetV2 for the demo. Its nine-photo Google diagnostic matched only 3/9 provisional labels, so whole-leaf presentation reliability remains unresolved. Tea, luffa and Mangalore cucumber proxy hosts limit applicability to common cucumber. See results/corrected/external_diagnostic.csv and deployed_test_report.txt.

Run in PowerShell from this directory with the project environment active. Put unlabelled photos directly in demo_input and run:

~~~powershell
python -m src.leaf_demo
~~~

Each photo receives five relative scores, a predicted class, a processed JPEG and a labelled stage panel in results/demo/<photo stem>. The default uses the DIP model selected by validation macro-F1. Use --model dip_mobilenet or --model dip_svm for an explicit comparison. SVM percentages are normalized decision scores. Both sets of scores express model preference and are uncalibrated.

Every demo run saves one Excel-readable predictions.csv in results/demo, or in the directory specified by --output. Each image has a row with the predicted class, all five scores in percent, model, segmentation decision and any processing error. Running without a photo argument refreshes the file for all current demo_input images. Close the CSV in Excel before rerunning on Windows.

**Rebuild.** Keep the original source datasets under external_data as described in [dataset construction](reports/dataset_construction.md). Install requirements.txt. Training uses the standard torchvision ImageNet MobileNetV2 weights and needs them cached locally or downloadable on first use. The demo uses local trained weights. Fetch the field subset before rebuilding:

~~~powershell
python -m src.fetch_field_data
python -m src.train_corrected
python -m src.evaluate_leaf_demo
python -m src.dip_stage_storage
python -m src.plot_corrected
~~~

The same sequence is available as `.\rebuild_dip.ps1`. A normal presentation uses only `python -m src.leaf_demo`.

For a new independently labelled field set, create a CSV with headers path,label,source. Paths are relative to the CSV, label is one of Healthy Leaf, Leaf Miner, Red Mite, Whitefly or Yellow Mosaic, and source identifies the original photo or verified collection. Run python -m src.evaluate_field_set field_cases.csv --output results/field_predictions.csv. The evaluator prints class metrics and saves every prediction and all five scores. It rejects exact benchmark duplicates. Related photos and label reliability require source review.

The new field data comes from [WMC-Leafset v4](https://data.mendeley.com/datasets/8m2ytxd4dg/4). Its [authors](https://pmc.ncbi.nlm.nih.gov/articles/PMC12664399/) document annotation 0 as leaf-miner damage and 1 as other pest damage. The fetcher verifies the publisher's SHA-256 digest for each cached file. The build consumes original tea images, luffa mosaic images, singly labelled Whitefly source frames, greenhouse Healthy Leaf images, and annotated Mangalore cucumber Leaf Miner and Healthy Leaf photos. Mangalore cucumber, tea and luffa are proxy hosts for the intended common cucumber field demonstration. Miner boxes provide labelled symptom views. A whole field frame is included as Leaf Miner only when all annotations are Leaf Miner. Crops inherit their parent photo's split.

The corrected manifest and measurements live in results/corrected. The input split is stratified by source with target proportions of 70/15/15 by group. Exact duplicates, known variants and visually similar source photos are grouped before splitting. Unknown plant or session identity can still connect photos across splits. A frozen ImageNet feature extractor and a newly fitted five-class linear head are trained for raw and DIP inputs. A separate RBF SVM uses HOG, LBP and HSV features. Weights give equal total training influence per class and group within class. Epoch selection and SVM tuning use validation macro-F1. Test metrics include both full frames and annotated views. The older reports and results/metrics describe the earlier split.

## Project owner

AP (Ayushman) is the sole owner of this Digital Image Processing course project.

## Folder Structure
*   `data/`: Datasets (Raw and Processed)
*   `src/`: Python scripts (DIP, ML, DL, Evaluation)
*   `models/`: Saved model weights
*   `results/`: Metrics, CSV files, and plots

## Environment Setup
1. Create virtual environment: `python -m venv .fresh-env`
2. Activate it: `.\.fresh-env\Scripts\Activate.ps1`
3. Install dependencies: `pip install -r requirements.txt`
<!-- 
external_data/: This holds the heavy, raw datasets (the Roboflow YOLO files, the tea dataset, the mosaic dataset). We keep this out of GitHub so it doesn't break the repository.

src/config.py: The "single source of truth." It holds the paths, image sizes (224x224), and class names so the project uses consistent settings.

src/crop_pests.py (The New File): Purpose: The Roboflow dataset gives us full images of leaves with tiny bounding boxes drawn around the bugs. Our DIP pipeline needs only the bug/lesion close-up to learn from. This script reads the YOLO .txt files, cuts out (crops) the Whitefly and Leaf Miner bugs, and saves them as individual .jpg files into data/raw/train/Whitefly/ and data/raw/train/Leaf_Miner/.

src/crop_yolo.py (The Old File): Purpose: This was an earlier draft. It is obsolete because we now have the exact class IDs from your data.yaml. You can safely delete this file. -->

## Corrected extra experiments

See [the experiment guide](results/corrected/extra_experiments/README.md) for the completed corrected-model studies, CSV outputs, plots, heatmaps and limitations. Reproduce them with rerun_extras.ps1.
