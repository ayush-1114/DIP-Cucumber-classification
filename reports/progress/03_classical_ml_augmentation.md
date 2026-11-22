# Progress 3: Classical ML Baseline and Targeted Augmentation

**DIP project maintainer:** AP · **Branch:** `classical-ml` · **Date:** 17 Nov 2026
**Code:** `src/classical_ml/` (features, augment, train)

## Summary

Before: the SVM and Random Forest, trained on the imbalanced data, almost never
recognised Leaf_Miner. After: augmenting the small classes in the training set raised
Random Forest macro-F1 from **0.75 to 0.86** and Leaf_Miner recall from **10% to 60%**.

## Setup (same for before and after)

- Data: `data/processed` from the DIP pipeline, shared split, **151 test images**
- Features: HOG + LBP + HSV colour histograms, 6208 values per image
- Models: SVM (scaler, PCA 95%, RBF) and Random Forest (300 trees), both
  `class_weight="balanced"`; settings chosen on validation by macro-F1
- The only difference: **after** adds augmented copies of the small classes to the
  training set (Leaf_Miner +167, Healthy +181; flips, rotations, brightness/contrast).
  Validation and test images are never augmented.

## Before vs after

| Model | Accuracy | Macro-F1 | Leaf_Miner recall | Healthy recall |
|---|---|---|---|---|
| SVM, before | 90.7% | 0.798 | 20% (2/10) | 83% (5/6) |
| SVM, after | 88.7% | 0.802 | 40% (4/10) | 100% (6/6) |
| Random Forest, before | 89.4% | 0.746 | 10% (1/10) | 83% (5/6) |
| **Random Forest, after** | **93.4%** | **0.861** | **60% (6/10)** | **100% (6/6)** |

Per-class F1, Random Forest:

| Class | Before | After |
|---|---|---|
| Whitefly | 0.89 | 0.92 |
| Red_Mite | 0.98 | 1.00 |
| Yellow_Mosaic | 0.94 | 0.97 |
| Leaf_Miner | **0.15** | **0.67** |
| Healthy | 0.77 | 0.75 |

## Why it is better

- **The rare class is finally detected:** Random Forest Leaf_Miner F1 went from 0.15 to
  0.67, the largest single improvement.
- **Macro-F1, not accuracy, shows it:** SVM accuracy actually dropped (90.7% to 88.7%)
  while its rare-class recall doubled. Accuracy hides the small classes; a model that
  never predicts Healthy or Leaf_Miner still scores about 89%.
- **Best baseline for the DL comparison:** Random Forest with augmentation, macro-F1 0.86.

## Trade-offs to report honestly

- **Healthy precision fell** (Random Forest 0.71 to 0.60): the model now predicts
  Healthy more often, catching all 6 but with more false alarms. Healthy F1 is roughly
  unchanged (0.77 to 0.75).
- **SVM gained little overall** (macro-F1 0.798 to 0.802); its Whitefly recall dropped
  (89% to 80%).
- **Tiny test sets:** Healthy has 6 test images and Leaf_Miner 10, so one image changes
  recall by 10-17 points.
- **Speed:** SVM about 35 ms per image, Random Forest about 90 ms (laptop, not idle;
  to be re-measured).
