"""Hand-crafted features for the classical ML track (roadmap Phase 3, Track 1).

Every image becomes one fixed-length vector made of three parts:

  HOG    - Histogram of Oriented Gradients: SHAPE / edge directions
           (leaf-miner tunnels, lesion outlines).
  LBP    - Local Binary Patterns: fine TEXTURE (mite speckling, mosaic mottling).
  Colour - HSV histograms: COLOUR (yellowing, healthy green).

Quick check on one image (from the project root, inside mlee-env):

    python -m src.classical_ml.features path/to/image.jpg
"""

import sys

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

from src import config

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# HOG: 224x224 image, 16x16-pixel cells, 2x2-cell blocks, 9 directions -> 6084 values
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP at two scales: (neighbours, radius). "uniform" LBP gives neighbours + 2 bins.
LBP_SCALES = ((8, 1), (16, 2))

# Colour: bins for the H, S and V channels
COLOR_BINS = (32, 32, 32)


def load_image(path):
    """Read an image as BGR and resize it to config.IMG_SIZE (224x224).

    The DIP-processed images are already 224x224, so for them the resize does nothing.
    np.fromfile + imdecode is used because cv2.imread fails on some Windows paths.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if (img.shape[1], img.shape[0]) != config.IMG_SIZE:
        img = cv2.resize(img, config.IMG_SIZE, interpolation=cv2.INTER_AREA)
    return img


def hog_features(gray):
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def lbp_features(gray):
    parts = []
    for n_points, radius in LBP_SCALES:
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        parts.append(hist / hist.sum())
    return np.concatenate(parts).astype(np.float32)


def color_features(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    parts = []
    # OpenCV ranges: H is 0-179, S and V are 0-255
    for channel, (bins, upper) in enumerate(zip(COLOR_BINS, (180, 256, 256))):
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, upper]).ravel()
        parts.append(hist / hist.sum())
    return np.concatenate(parts).astype(np.float32)


def extract_features(bgr):
    """BGR image (224x224) -> 1-D float32 vector [HOG | LBP | colour]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.concatenate([hog_features(gray), lbp_features(gray), color_features(bgr)])


def list_images(split_dir):
    """Return (paths, labels) for split_dir/<Class>/*, labels from config.CLASS_TO_IDX."""
    paths, labels = [], []
    for cls in config.CLASSES:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            continue
        for path in sorted(cls_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMG_EXTS:
                paths.append(path)
                labels.append(config.CLASS_TO_IDX[cls])
    return paths, labels


def extract_folder(split_dir, desc=None):
    """Extract features for every image in split_dir/<Class>/. Returns X, y, paths."""
    paths, labels = list_images(split_dir)
    X = [extract_features(load_image(p)) for p in tqdm(paths, desc=desc or split_dir.name)]
    if not X:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64), []
    return np.stack(X), np.array(labels, dtype=np.int64), paths


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    image = load_image(sys.argv[1])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"HOG:    {hog_features(gray_image).shape[0]} values")
    print(f"LBP:    {lbp_features(gray_image).shape[0]} values")
    print(f"Colour: {color_features(image).shape[0]} values")
    print(f"Total:  {extract_features(image).shape[0]} values")
