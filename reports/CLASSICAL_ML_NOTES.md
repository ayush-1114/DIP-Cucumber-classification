# What Was Pushed: Classical ML

Branch `classical-ml`, commit `902ffe6`, based on the latest `main` (`4f499ad`,
the project's PR #3). Pushed 17 Nov 2026. `main` is unchanged; the branch goes in
through a pull request that Ayushman reviews and merges.

Pull request link:
https://github.com/ayush-1114/Cucumber-Disease-Pipeline/pull/new/classical-ml

## Files pushed

| File | New / changed | What it is |
|---|---|---|
| `src/classical_ml/features.py` | New | HOG (shape) + LBP (texture) + HSV colour histograms: 6208 numbers per image |
| `src/classical_ml/augment.py` | New | Extra copies of the small classes, train split only |
| `src/classical_ml/train.py` | New | Trains SVM and Random Forest, picks settings on val, scores once on test, saves models, predictions and summary |
| `src/classical_ml/__init__.py` | New | Makes the folder a Python package |
| `src/classical_ml/README.md` | New | How to run the scripts and what they output |
| `reports/CODE_WALKTHROUGH.md` | New | Plain-language explanation of every script, for presentation prep |
| `results/metrics/classical_ml_processed_summary.json` | New | Scores, chosen settings, per-class results, timing (with augmentation) |
| `results/metrics/svm_processed_test_predictions.csv` | New | One row per test image for the SVM: filename, true class, predicted class |
| `results/metrics/rf_processed_test_predictions.csv` | New | Same for the Random Forest |
| `results/metrics/*_processed_noaug_*` (3 files) | New | Same run without augmentation, for comparison |
| `results/figures/augmentation_examples_processed.png` | New | Picture of what augmentation does, for the report |
| `requirements.txt` | Changed | Added `scikit-image` (needed for HOG and LBP) |
| `.gitignore` | Changed | Added `data/features/` (cached features, local only) |

Not pushed (Git ignores them): the trained models `models/*.pkl`, the images and
the cached features. Anyone can recreate them by running the scripts.

## How the results were made

1. Started from the latest `main` (the project's PR #3, updated DIP pipeline).
2. Rebuilt `data/processed` with his updated `src/dip/pipeline.py`.
3. Checked it against the shared list: 702 train / 151 val / 151 test, nothing to
   move, so the raw and processed sets use the same images.
4. Trained both models with and without augmentation.

## Results (151 test images, processed data)

| Model | Augmentation | Accuracy | Macro-F1 | Leaf_Miner found | Healthy found | ms / image |
|---|---|---|---|---|---|---|
| SVM | No | 90.7% | 0.80 | 2 of 10 | 5 of 6 | 36 |
| SVM | Yes | 88.7% | 0.80 | 4 of 10 | 6 of 6 | 35 |
| Random Forest | No | 89.4% | 0.75 | 1 of 10 | 5 of 6 | 93 |
| **Random Forest** | **Yes** | **93.4%** | **0.86** | **6 of 10** | **6 of 6** | **87** |

- Best model: **Random Forest with augmentation**, macro-F1 0.86.
- Augmentation helps most on Leaf_Miner (1 to 6 of 10 found for Random Forest).
- Red_Mite is 100% for every model.
- The SVM is about 2.5 times faster per image.

## Caveats

- Healthy (6) and Leaf_Miner (10) test sets are tiny, so their numbers are rough.
- Timing was measured on a laptop with other work running; re-measure on an idle
  machine before the final speed comparison.
- `svm_model.pkl` and `rf_model.pkl` are the augmented versions. They are not in
  Git. Keep local copies available for evaluation.

## Setup after merging

```
git pull origin main
pip install -r requirements.txt
python -m src.classical_ml.train --no-cache
```
