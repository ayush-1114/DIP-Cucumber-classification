# Dataset Construction and Split (SN)

Notes for the report on how the 5-class dataset was assembled, cleaned and split.
Numbers come from `results/metrics/dataset_split_summary.csv`, written by
`src/data_prep/split_dataset.py`.

## Sources

No single public dataset covers all five classes, so the dataset combines several
sources (links are in `roadmap- MLEE project.pdf`).

| Class | Source | How the images were obtained |
|---|---|---|
| Whitefly | Roboflow "Pests in Cucumber Plants" (YOLO format) | Bounding boxes cropped by `src/crop_pests.py` |
| Leaf_Miner | Roboflow "Pests in Cucumber Plants" (YOLO format) | Bounding boxes cropped by `src/crop_pests.py` |
| Red_Mite | Mendeley Indian Tea Leaves, "Red Spider" folder | Proxy class (tea leaves); capped at 1500 by `src/organize_datasets.py` |
| Yellow_Mosaic | Mendeley Luffa aegyptiaca, "Cucumber Mosaic Virus" folder | Proxy class; capped at 1500 |
| Healthy | Kaggle Cucumber Greenhouse dataset, "Healthy Leaves" folder | Copied as-is |

## Cleaning and downsampling

- **Exact duplicates dropped: 46 files** (same bytes, found by MD5 hash).
  41 were in Healthy (every Healthy image was present twice under two names) and
  5 in Yellow_Mosaic.
- **Downsampled to at most 300 images per class** (team decision, to reduce the
  imbalance and training time), chosen with the shared seed. This removed 1200
  Red_Mite, 1197 Yellow_Mosaic and 11 Whitefly images.
- Images that are not used are moved to `_unused/`, never deleted.
- After both steps: **1004 images**.

## Stratified 70/15/15 split

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Whitefly | 210 | 45 | 45 | 300 |
| Red_Mite | 210 | 45 | 45 | 300 |
| Yellow_Mosaic | 210 | 45 | 45 | 300 |
| Leaf_Miner | 43 | 10 | 10 | 63 |
| Healthy | 29 | 6 | 6 | 41 |
| **Total** | **702** | **151** | **151** | **1004** |

- **One shared list:** the split is defined by `data/split_manifest.csv`, which is
  committed. `python -m src.data_prep.split_dataset` applies that list to both
  `data/raw` and `data/processed`, so the classical ML, deep learning and evaluation
  all use the same train, val and test images.
- **Stratified:** the split is done class by class, so every class keeps the 70/15/15 ratio.
- **Leakage-safe:** the Whitefly crops come from only 65 photos and the 63 Leaf_Miner
  crops from only 18 photos. All crops of one photo are kept in the same split;
  otherwise near-identical crops would sit in both train and test and inflate the score.
- **Reproducible:** fixed seed (`SEED = 42` in `src/config.py`) and sorted file names,
  so rebuilding the list from the same `raw.zip` gives the identical list on any machine.

## Class imbalance

After downsampling, the largest classes (300) are still about 7 times the smallest (Healthy, 41).

- Classical models use `class_weight="balanced"`, and the small classes are augmented
  in the training set only (`src/classical_ml/augment.py`).
- Models are compared with **macro-F1**, not accuracy. A model that never predicts
  Healthy or Leaf_Miner would still reach about 89% accuracy.

## Known limitations (to state in the report)

- **Healthy (41) and Leaf_Miner (63) are very small.** Their test sets have 6 and 10
  images, so one mistake moves per-class recall by 10-17 percentage points.
- **Each class comes from a different source**, with different crops, backgrounds and
  image sizes (Healthy: large photos on a white background; Whitefly / Leaf_Miner: small
  crops; Red_Mite: tea leaves, about a fifth of them black-and-white thresholded images). A model can partly separate classes by source style
  rather than by disease. Grad-CAM (Phase 4.3) is the planned check for this.
- **Red_Mite and Yellow_Mosaic are proxy classes** from other plants (tea, luffa).
- Only exact duplicates were removed; near-duplicates (for example augmented copies
  inside a source dataset) were not checked.
