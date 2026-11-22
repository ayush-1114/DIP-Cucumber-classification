# Progress 1: One Shared Train/Val/Test Split

**DIP project maintainer:** AP · **Merged in:** PR #2 (`fix-shared-split`) · **Date:** 16 Nov 2026
**Code:** `src/data_prep/split_dataset.py`, `data/split_manifest.csv`

## Summary

Before: there was no train/val/test split, the dataset contained hidden duplicates,
and each installation would have split the data differently. After: one committed list
gives every installation the same 702 / 151 / 151 split, with no duplicates and no leakage.

## Before

| Problem | Evidence |
|---|---|
| No split at all | `data/raw/val` and `data/raw/test` were empty; all 3458 images sat in `train/` |
| Hidden duplicates | 46 files were byte-identical copies (MD5 hash). **Every** Healthy image was there twice: 82 files, only 41 real images. 5 more in Yellow_Mosaic |
| Each person would split differently | The suggested fix was each person calling `train_test_split` on their own. Different seeds or file orders give different test sets, so SVM and MobileNetV2 scores could not be compared |
| Downsampling not repeatable | `src/downsample_data.py` shuffled files in the order the OS listed them, so "the 300 images kept" differed between laptops |
| Data leakage risk | Whitefly: 311 crops from only **65** photos. Leaf_Miner: 63 crops from only **18** photos. A per-image split puts crops of the same photo in both train and test |
| First attempt at grouping skewed the ratios | Grouping crops with `train_test_split` gave Whitefly **167 / 71 / 73 (54% / 23% / 23%)** instead of 70 / 15 / 15 |

## After

| Fix | Result |
|---|---|
| One committed list (`data/split_manifest.csv`) | Every installation runs the same script and gets identical folders |
| Applies to both raw and processed | Verified: `data/raw` and `data/processed` both 702 / 151 / 151, same images |
| Duplicates removed | 3458 files to 3412 unique images (moved to `_unused/`, not deleted) |
| Repeatable downsampling to 300 per class | Same images kept on every machine (seeded, sorted file names) |
| Crops of one photo kept together | **0** photos split across train/val/test (checked on all groups) |
| Size-aware group assignment | Whitefly now **210 / 45 / 45 (70 / 15 / 15)** |
| Repeatable | Rebuilding the list twice gives a byte-identical file |

Final split:

| Class | Train | Val | Test |
|---|---|---|---|
| Whitefly | 210 | 45 | 45 |
| Red_Mite | 210 | 45 | 45 |
| Yellow_Mosaic | 210 | 45 | 45 |
| Leaf_Miner | 43 | 10 | 10 |
| Healthy | 29 | 6 | 6 |
| **Total** | **702** | **151** | **151** |

## Why it is better

- **Fair comparison:** the classical ML, the deep learning and the evaluation all use the
  same test images, so their F1-scores can be compared directly.
- **Honest scores:** no near-identical crop can be in both train and test.
- **Correct class sizes:** Healthy is reported as 41 images, not 82.

## Limitation still open

Healthy (6) and Leaf_Miner (10) have very small test sets, so their scores are rough.
More real images for these two classes would help more than any split change.
