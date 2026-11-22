# Code Walkthrough (presentation prep)

A plain-language guide to the four scripts in the classical ML track: what each one
does, why it is written that way, and the questions you are likely to be asked.

All commands are run from the project root with `mlee-env` activated.

## The big picture

```
data/raw/train (everything)                         src/config.py
        |                                           (paths, 224x224, SEED=42,
        v                                            70/15/15, 5 class names)
split_dataset.py  --->  data/raw/train | val | test
        |
        |   (the DIP pipeline turns data/raw into data/processed = "Dataset B")
        v
features.py   image -> 6208 numbers (HOG + LBP + colour)
augment.py    extra copies of the small classes, TRAIN only
        |
        v
train.py      SVM + Random Forest -> models/, results/metrics/ (for the evaluation)
```

`src/config.py` is the shared settings file. Every script imports it, so paths, the
image size, the seed and the class names are defined in exactly one place.

---

## 1. `src/data_prep/split_dataset.py` - the 70/15/15 split

**Run:** `python -m src.data_prep.split_dataset` (add `--dry-run` to preview only)

**The idea:** the split is one committed list, `data/split_manifest.csv`. It names
every image we use and says train, val or test. Everyone applies the same list, so
the SVM, the CNN and the evaluation all use identical images. Without this, each
person would make their own random split and the F1-scores could not be compared.

**Applying the list (the normal run)**

- For `data/raw` and `data/processed`, each image is matched to the list by class and
  file name and moved to `train/`, `val/` or `test/`.
- Images not in the list go to `_unused/`. Nothing is deleted.

**Building the list (`--rebuild`, performed once)**

1. **Collect** every raw image.
2. **Drop exact duplicates** using an MD5 hash (a fingerprint of the file's bytes).
   It found 46: every Healthy image was stored twice (41), plus 5 in Yellow_Mosaic.
3. **Downsample** each class to at most 300 images, with the shared seed.
4. **Group crops of the same photo.** Whitefly and Leaf_Miner images are crops cut
   out of Roboflow photos (63 Leaf_Miner crops from 18 photos). Crops of one photo
   are treated as one group.
5. **Split class by class** into 70% train, 15% val, 15% test, keeping each group
   whole. Groups are shuffled with the seed, then given one at a time (biggest first)
   to whichever split is furthest below its target.

Result: 702 train, 151 val, 151 test.

**Key terms**

- **Stratified:** each class is split separately, so every class has the same
  70/15/15 proportions. A plain random split could put all 41 Healthy images in train
  and none in test.
- **Data leakage:** information from the test set sneaking into training. Two crops
  of the same photo look almost identical; if one is in train and one in test, the
  model is tested on something it has effectively seen, and the score is inflated.
- **Seed:** the starting number for the random generator. Same seed, same shuffle,
  so the split is repeatable on any laptop.
- **Train / val / test:** train is for fitting the model, val is for choosing
  settings, test is touched once at the end for the final score.

**Likely questions**

- *Why not sklearn's `train_test_split(stratify=...)`?* It splits individual images,
  so it cannot keep crops of one photo together. With the groups it also drifted to
  54/23/23 for Whitefly, because groups have different sizes.
- *Why move duplicates instead of deleting them?* So the step can be undone and
  checked.
- *How is the same split reproduced?* They pull the committed manifest and run
  the script, which applies it to their raw and processed folders.

---

## 2. `src/classical_ml/features.py` - turning an image into numbers

**Run a check:** `python -m src.classical_ml.features path/to/image.jpg`

An SVM or Random Forest cannot read an image directly; it needs a fixed-length list
of numbers. Each image is resized to 224x224 and described by three feature types,
joined into one vector of **6208 values**.

| Feature | What it measures | Size | Why it helps here |
|---|---|---|---|
| **HOG** (Histogram of Oriented Gradients) | Shape: which edge directions occur where | 6084 | Leaf-miner tunnels, lesion outlines, leaf edges |
| **LBP** (Local Binary Patterns) | Fine texture | 28 | Mite speckling, mosaic mottling |
| **Colour histogram** (HSV) | How much of each colour | 96 | Yellowing vs healthy green |

**How each works**

- **HOG:** convert to grey, measure the edge direction at every pixel, then for each
  16x16-pixel cell count how many edges point in each of 9 directions. Cells are
  normalised in 2x2 blocks so lighting changes matter less.
  Size: 13 x 13 block positions x 4 cells x 9 directions = 6084.
- **LBP:** for each pixel, compare it with its neighbours on a circle: brighter = 1,
  darker = 0. That pattern of 0s and 1s is a code for the local texture. The feature
  is a histogram of the codes over the image. Two scales are used: 8 neighbours at
  radius 1 (10 bins) and 16 neighbours at radius 2 (18 bins) = 28.
  "Uniform" LBP only keeps the common simple patterns, which keeps it short.
- **Colour:** convert to HSV (Hue = which colour, Saturation = how strong,
  Value = how bright) and make a 32-bin histogram of each channel = 96. HSV is used
  instead of RGB because hue stays roughly the same when lighting changes.

**Likely questions**

- *Why hand-crafted features?* That is the point of Track 1: classical feature
  engineering (we choose the features) versus deep learning (the CNN learns its own).
- *Why 224x224?* It is `IMG_SIZE` in the config, the input size MobileNetV2 and
  ResNet50 expect, so both tracks see the same images.
- *HOG is 6084 of 6208 values. Does it drown out the others?* Partly, which is why
  the SVM pipeline standardises every feature and applies PCA first (see below).

---

## 3. `src/classical_ml/augment.py` - helping the small classes

**Save example picture:** `python -m src.classical_ml.augment`
(writes `results/figures/augmentation_examples_raw.png`)

**The problem:** even after downsampling, train has 210 images for each big class
but only 29 Healthy and 43 Leaf_Miner. The models learn to ignore the small classes.

**The fix:** every smaller class is topped up to the size of the largest class (210)
with randomly changed copies of its own training images:

- left-right and up-down flips
- a turn of 0 / 90 / 180 / 270 degrees, plus a small tilt of up to 25 degrees
- contrast x0.8 to x1.2, brightness -20 to +20

These are changes that do not alter the label: a flipped leaf-miner leaf is still a
leaf-miner leaf.

**The rules that keep it honest**

- **Train only.** Val and test are never augmented, so scores are measured on real
  images.
- **After the split.** Copies are made from train images only, so no copy of a test
  image can end up in train.
- **In memory.** Nothing is written into `data/`, so the split and the pipeline are
  unaffected.

**Likely questions**

- *Does augmentation add new information?* No. It shows the same leaves in new
  poses, which stops the model memorising orientation and lighting. It is not a
  substitute for collecting more real images.
- *Why also use `class_weight="balanced"`?* They do different jobs. Class weights
  make mistakes on rare classes cost more; augmentation gives the model more varied
  examples of them.

---

## 4. `src/classical_ml/train.py` - SVM and Random Forest

**Run:** `python -m src.classical_ml.train` (Dataset B, the real run)
Options: `--dataset raw` (code check only), `--no-augment` (to compare), `--no-cache`.

**What it does, in order**

1. **Extracts features** for train, val and test, and caches them in
   `data/features/` so the next run is fast. If the split or the processed images
   change, run with `--no-cache`.
2. **Adds the augmented copies** to train.
3. **Tries a few settings for each model.** Each setting is fitted on train and
   scored on val; the best one is kept.
4. **Scores the chosen model once on test.**
5. **Saves** the model (`models/svm_model.pkl`, `models/rf_model.pkl`), the test predictions and a summary (scores, chosen settings,
   milliseconds per image) for evaluation.

**The two models**

- **SVM (Support Vector Machine)** - finds the boundary that separates the classes
  with the widest margin. The pipeline is:
  1. `StandardScaler`: puts every feature on the same scale (SVMs are sensitive to
     scale).
  2. `PCA` keeping 95% of the variance: compresses 6208 values to far fewer,
     removing redundancy. Fewer dimensions = faster and less overfitting.
  3. `SVC` with an RBF kernel: the kernel lets the boundary be curved.
  The setting tried is **C** (0.1, 1, 10, 100): small C = smoother boundary that
  tolerates some training mistakes; large C = fits the training data more tightly.
- **Random Forest** - 300 decision trees, each trained on a random sample of the
  data and features; the forest takes a vote. No scaling needed. The setting tried
  is **max_depth** (how many questions a tree may ask): unlimited, 10 or 20.

**Key terms**

- **Macro-F1:** F1 combines precision (of the images I called X, how many were X)
  and recall (of the real X images, how many I found). Macro-F1 is the plain average
  of F1 over the 5 classes, so Healthy counts as much as Red_Mite.
- **Why not accuracy?** A model that never predicts Healthy or Leaf_Miner still gets
  about 89% accuracy, because those classes are only 16 of 151 test images.
- **Hyperparameter:** a setting we choose (C, max_depth), as opposed to what the
  model learns from data. Choosing it on val, not test, keeps the test score honest.

**Likely questions**

- *Why must the final run use Dataset B?* The project compares classical ML against
  deep learning on the same DIP-processed images. Features from raw images would
  make that comparison invalid.
- *Why is inference timed?* The analysis compares speed (ms per image) of classical ML versus
  deep learning for edge deployment. The timing covers the whole path: read the
  image, extract features, predict.
- *Why is the SVM "one model" when there are 5 classes?* sklearn's SVC trains one
  SVM per pair of classes (10 pairs) and takes a vote.

---

## Honest limitations (say these before you are asked)

- Healthy (41 images) and Leaf_Miner (63) are tiny. Their test sets have 6 and 10
  images, so per-class scores for them are rough.
- Each class comes from a different source and looks different in style, so a model
  can partly tell classes apart by source rather than by disease. For example, roughly a fifth of the
  Red_Mite images (32 of a 150-image sample) are black-and-white thresholded images,
  which almost never occurs in the other classes. Grad-CAM is the planned check.
- Red_Mite and Yellow_Mosaic are proxy classes from other plants (tea, luffa).
