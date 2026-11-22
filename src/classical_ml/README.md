# Classical ML Track

Track 1 of the project: hand-crafted features + SVM and Random Forest. This is the
baseline that the deep-learning track (MobileNetV2 / ResNet50) is compared against.

## Files

| File | What it does |
|---|---|
| `features.py` | Turns a 224x224 image into 6208 numbers: HOG (shape, 6084), LBP (texture, 28) and HSV colour histograms (96). |
| `augment.py` | Tops up the small classes in the TRAIN split only (flips, 90-degree turns, small tilt, brightness/contrast) until each class matches the largest one. Val and test are never touched, and nothing is written to `data/`. |
| `train.py` | Extracts features, picks settings on VAL by macro-F1, scores once on TEST, and saves the models, predictions and a summary. |

## How to run

From the project root, with `mlee-env` active:

```
pip install -r requirements.txt                 # scikit-image is new
python -m src.dip.pipeline                      # makes data/processed/train|val|test
python -m src.data_prep.split_dataset           # makes sure raw and processed match the shared list
python -m src.classical_ml.train --no-cache     # the real run, on data/processed
```

Options for `train.py`:

| Option | Effect |
|---|---|
| `--dataset raw` | Train on raw images instead (code check only; not comparable with the DL track) |
| `--no-augment` | Skip augmentation, to compare with and without |
| `--no-cache` | Re-extract features. Use it whenever the images or the split change |

## Outputs

| File | For whom |
|---|---|
| `models/svm_model.pkl`, `models/rf_model.pkl` | The trained models (local only; Git ignores `*.pkl`) |
| `results/metrics/svm_processed_test_predictions.csv`, `rf_processed_test_predictions.csv` | the evaluation: one row per test image, `filename, y_true, y_pred`, as class names |
| `results/metrics/classical_ml_processed_summary.json` | the evaluation: chosen settings, val and test scores, per-class precision/recall/F1, ms per image |
| `results/figures/augmentation_examples_processed.png` | Report: what the augmentation does |

The `*_noaug_*` files are the same run without augmentation, for comparison.

## Models

- **SVM:** StandardScaler -> PCA (95% of variance) -> RBF SVM, `class_weight="balanced"`.
  C tried: 0.1, 1, 10, 100.
- **Random Forest:** 300 trees, `class_weight="balanced"`. max_depth tried: none, 10, 20.

Models are chosen and compared by **macro-F1**, because accuracy hides the small
classes: a model that never predicts Healthy or Leaf_Miner still gets about 89%.

## Notes

- Inference time covers reading the image, extracting features and predicting.
  Measure it on an otherwise idle machine; background work inflates it.
- `config.CLASSES` order is not alphabetical, while `torchvision`'s `ImageFolder`
  sorts classes alphabetically. The prediction CSVs use class names, so compare by
  name, not by index.
- Healthy (6) and Leaf_Miner (10) have very few test images, so their per-class
  scores are rough.

See `reports/CODE_WALKTHROUGH.md` for a plain-language explanation of each step.
