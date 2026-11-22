# Dataset Construction and Split

Notes for the report on how the 5-class dataset was assembled, cleaned and split.
Numbers come from `results/metrics/dataset_split_summary.csv`, written by
`src/data_prep/split_dataset.py`.

## Sources

The current dataset combines the sources supplied by the project. These sources provide different hosts and image types. Their transfer to whole cucumber leaves requires separate evaluation.

| Class | Source | How the images were obtained |
|---|---|---|
| Whitefly | [Roboflow "Pests in Cucumber Plants"](https://universe.roboflow.com/cpa-challenge/pests-in-cucumber-plants-kcmdr) (YOLO format) | Bounding boxes cropped by `src/crop_pests.py` |
| Leaf_Miner | [Roboflow "Pests in Cucumber Plants"](https://universe.roboflow.com/cpa-challenge/pests-in-cucumber-plants-kcmdr) (YOLO format) | Bounding boxes cropped by `src/crop_pests.py` |
| Red_Mite | [Mendeley Indian Tea Leaves, version 1](https://data.mendeley.com/datasets/jm378tjnrt/1), "Red Spider" folder | Proxy class (tea leaves); capped at 1500 by `src/organize_datasets.py` |
| Yellow_Mosaic | [Mendeley Luffa aegyptiaca, version 3](https://data.mendeley.com/datasets/dkptcnjn42/3), "Cucumber Mosaic Virus" folder | Proxy class; capped at 1500 |
| Healthy | [Kaggle Cucumber Greenhouse dataset](https://www.kaggle.com/datasets/farahseifeld/greenhouse-cucumber-growth-stages), "Healthy Leaves" folder | Copied as-is |

## Cleaning and downsampling

- **Exact duplicates dropped: 46 files** (same bytes, found by MD5 hash).
  41 were in Healthy (every Healthy image was present twice under two names) and
  5 in Yellow_Mosaic.
- **Downsampled to at most 300 images per class** (configuration choice, to reduce the
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
- **Roboflow source grouping:** the Whitefly crops come from only 65 photos and the 63 Leaf_Miner
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


## Source audit on 22 November 2026

The local Roboflow export lists five object categories in data.yaml. The crop script correctly selects IDs 3 (Serpentine leafminer) and 4 (Whitefly). Counting local annotation files found 20 images with 77 Leaf Miner boxes and 67 images with 327 Whitefly boxes before the project's cleaning and splitting. These counts describe the locally available annotations. The export README states 563 images, so annotation counts should not be treated as a complete upstream inventory.

Visual inspection of one example per selected class shows a small insect inside each box. The crop script adds 15 pixels around the box. In particular, the inspected Leaf Miner example supplies insect appearance while the new demonstration photos show winding leaf mines. All selected annotations need review before constructing whole-leaf classification labels.

The tea dataset publisher describes ten transformed variants per original, including binary and grayscale images, four blur levels, two noise levels and two rotations. Local filenames preserve an IMG source identifier, but organize_datasets.py replaces these with sequential filenames. The current split groups Roboflow crops while treating each renamed tea image as its own group. Exact duplicate removal cannot remove these transformed relatives. See the local source-group audit for measured overlap.

The Mendeley tea page was readable and confirms the proxy host and transformations. The Luffa page timed out, the Kaggle page exposed no readable description, and the supplied Roboflow page could not be retrieved through the web tool. Their links are recorded as supplied provenance. The local export metadata and files provide the Roboflow annotation evidence.

The statement that no public cucumber-specific Red Mite dataset exists has not been established by this audit. The justified statement is that the current project uses a tea-leaf proxy.


### Confirmed tea source overlap

Byte-hash matching mapped all 300 current Red Mite files back to the supplied tea dataset. They originate from 126 source-photo IDs. Transformed versions of 55 source IDs cross split boundaries, including 34 source IDs shared between training and test. The historical split protects Roboflow crop groups but permits source-related tea images across subsets. The old Red Mite test scores need replacement after a source-group rebuild and retraining. [Measured groups](../results/metrics/tea_source_group_audit.json)

For a corrected dataset, recover original source IDs before renaming, group all transformed versions with their original, and apply augmentation only to training data. Rebuild both raw and processed variants with the same group assignment and regenerate metrics. Select a whole-leaf dataset based on the actual symptom/class definition.
