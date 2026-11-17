"""Targeted augmentation for the small classes (classical ML track).

Problem: Healthy and Leaf_Miner have only a few dozen training images, while
Red_Mite and Yellow_Mosaic have about 1000 each. The models then mostly ignore
the small classes.

Fix: every class with fewer training images than the largest class gets extra
copies of its TRAINING images with random flips, rotations and brightness/contrast
changes, until it is as big as the largest class.

Rules that keep the evaluation honest:
  - Only the TRAIN split is augmented. Val and test images are never touched.
  - Nothing is written into data/. The copies are made in memory during training,
    so the split and AP's DIP pipeline are not affected.
  - The random generator uses the shared SEED, so results are repeatable.

Save a picture of example augmentations (for the report):

    python -m src.classical_ml.augment            # uses data/raw/train
    python -m src.classical_ml.augment processed  # uses data/processed/train
"""

import sys

import cv2
import numpy as np
from tqdm import tqdm

from src import config
from src.classical_ml.features import extract_features, list_images, load_image

MAX_ROTATION_DEG = 25       # small random tilt, on top of random 90-degree turns
CONTRAST_RANGE = (0.8, 1.2)
BRIGHTNESS_RANGE = (-20, 20)


def augment_image(bgr, rng):
    """Return one randomly changed copy of a BGR image (same size as the input)."""
    out = bgr
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)                      # left-right flip
    if rng.random() < 0.5:
        out = cv2.flip(out, 0)                      # up-down flip
    out = np.rot90(out, k=int(rng.integers(0, 4)))  # 0 / 90 / 180 / 270 degrees
    out = np.ascontiguousarray(out)

    h, w = out.shape[:2]
    angle = float(rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # BORDER_REFLECT fills the corners with mirrored leaf instead of black triangles
    out = cv2.warpAffine(out, matrix, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    contrast = float(rng.uniform(*CONTRAST_RANGE))
    brightness = float(rng.uniform(*BRIGHTNESS_RANGE))
    return cv2.convertScaleAbs(out, alpha=contrast, beta=brightness)


def augmented_training_features(train_dir):
    """Features of the extra (augmented) training images.

    Returns X_extra, y_extra and a dict {class name: number of copies added}.
    """
    paths, labels = list_images(train_dir)
    rng = np.random.default_rng(config.SEED)
    X_extra, y_extra, added = [], [], {}
    target = max(labels.count(idx) for idx in set(labels)) if labels else 0

    for cls in config.CLASSES:
        idx = config.CLASS_TO_IDX[cls]
        cls_paths = [p for p, label in zip(paths, labels) if label == idx]
        n_missing = target - len(cls_paths)
        if not cls_paths or n_missing <= 0:
            continue

        images = [load_image(p) for p in cls_paths]   # read each original once
        for i in tqdm(range(n_missing), desc=f"augment {cls}"):
            source = images[i % len(images)]          # cycle through the originals
            X_extra.append(extract_features(augment_image(source, rng)))
            y_extra.append(idx)
        added[cls] = n_missing

    if not X_extra:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64), added
    return np.stack(X_extra), np.array(y_extra, dtype=np.int64), added


def save_examples(train_dir, out_path, n_copies=5):
    """Save a grid: one original per class on the left, then n_copies augmentations."""
    paths, labels = list_images(train_dir)
    rng = np.random.default_rng(config.SEED)
    rows = []
    for cls in config.CLASSES:
        cls_paths = [p for p, label in zip(paths, labels) if label == config.CLASS_TO_IDX[cls]]
        if not cls_paths:
            continue
        original = load_image(cls_paths[0])
        row = [original] + [augment_image(original, rng) for _ in range(n_copies)]
        rows.append(np.hstack(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    print(f"Saved {out_path} (rows: {', '.join(config.CLASSES)}; first column = original)")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "raw"
    base = {"raw": config.RAW_DIR, "processed": config.PROCESSED_DIR}[dataset]
    save_examples(base / "train",
                  config.RESULTS_DIR / "figures" / f"augmentation_examples_{dataset}.png")
