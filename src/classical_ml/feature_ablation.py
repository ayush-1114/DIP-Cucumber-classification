"""Run the seven Random Forest feature subsets on the shared DIP split.

From the repository root: python -m src.classical_ml.feature_ablation
"""

import csv
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, recall_score

from src import config
from src.classical_ml.augment import augmented_training_features
from src.classical_ml.features import (
    color_features,
    extract_folder,
    hog_features,
    lbp_features,
    list_images,
)
from src.classical_ml.train import build_rf, macro_f1


def feature_columns():
    width, height = config.IMG_SIZE
    gray = np.zeros((height, width), dtype=np.uint8)
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    sizes = (hog_features(gray).size, lbp_features(gray).size, color_features(bgr).size)
    edges = np.cumsum((0, *sizes))
    return {
        name: np.arange(edges[index], edges[index + 1])
        for index, name in enumerate(("HOG", "LBP", "Colour"))
    }


def main():
    root = config.PROCESSED_DIR
    if os.name == "nt" and root.drive and not str(root).startswith("\\\\"):
        root = Path("\\\\?\\" + str(root))

    with (config.ROOT_DIR / "data" / "split_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))

    splits = {}
    for split in config.SPLITS:
        expected = [(row["class"], row["filename"]) for row in manifest if row["split"] == split]
        paths, _ = list_images(root / split)
        actual = {(path.parent.name, path.name) for path in paths}
        if len(expected) != len(paths) or set(expected) != actual:
            raise ValueError(f"Processed {split} files differ from the shared manifest")
        features, labels, _ = extract_folder(root / split, desc=f"processed/{split}")
        splits[split] = (features, labels)

    train_x, train_y = splits["train"]
    extra_x, extra_y, added = augmented_training_features(root / "train")
    if len(extra_y):
        train_x = np.concatenate((train_x, extra_x))
        train_y = np.concatenate((train_y, extra_y))
    print(f"Training augmentation: {added}", flush=True)

    blocks = feature_columns()
    if sum(map(len, blocks.values())) != train_x.shape[1]:
        raise ValueError("Feature columns do not match the extracted vector")

    results = []
    for count in range(1, len(blocks) + 1):
        for subset in combinations(blocks, count):
            cols = np.concatenate([blocks[name] for name in subset])
            model = build_rf(max_depth=None).fit(train_x[:, cols], train_y)
            result = {"features": " + ".join(subset), "dimensions": len(cols)}
            for split in ("val", "test"):
                features, labels = splits[split]
                predictions = model.predict(features[:, cols])
                result[f"{split}_accuracy"] = accuracy_score(labels, predictions)
                result[f"{split}_macro_f1"] = macro_f1(labels, predictions)
                recalls = recall_score(
                    labels, predictions, labels=range(len(config.CLASSES)),
                    average=None, zero_division=0,
                )
                for name, recall in zip(config.CLASSES, recalls):
                    result[f"{split}_recall_{name}"] = recall
            print(f"{result['features']}: val macro-F1 {result['val_macro_f1']:.4f}", flush=True)
            results.append(result)

    output = config.RESULTS_DIR / "metrics" / "rf_feature_ablation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
