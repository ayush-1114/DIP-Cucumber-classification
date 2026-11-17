"""Measure how Leaf Miner training coverage affects the MobileNetV2 embedding SVM."""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import models

from src import config
from src.classical_ml.features import list_images
from src.classical_ml.mobilenet_svm import embed
from src.classical_ml.train import build_svm, macro_f1


FRACTIONS = (0.25, 0.50, 0.75, 1.0)
SEEDS = (42, 43, 44, 45, 46)
SVM_C = 1
LEAF_MINER = config.CLASS_TO_IDX["Leaf_Miner"]


def load_split(root, manifest, split):
    rows = [row for row in manifest if row["split"] == split]
    groups = {(row["class"], row["filename"]): row["group"] for row in rows}
    paths, labels = list_images(root / split)
    keys = [(path.parent.name, path.name) for path in paths]
    if not keys or len(keys) != len(rows) or len(groups) != len(rows) or set(keys) != set(groups):
        raise ValueError(f"Processed {split} files differ from the shared manifest")
    if any(not group.strip() for group in groups.values()):
        raise ValueError(f"Missing source group in {split} manifest")
    return paths, np.array(labels), np.array([groups[key] for key in keys])


def plot_curve(results, output):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    counts = [next(row["leaf_miner_train_groups"] for row in results
                   if row["target_fraction"] == fraction) for fraction in FRACTIONS]
    for metric, label in (("val_leaf_miner_recall", "Leaf Miner recall"),
                          ("val_macro_f1", "Macro-F1")):
        values = [np.array([row[metric] for row in results if row["target_fraction"] == fraction])
                  for fraction in FRACTIONS]
        ax.errorbar(counts, [scores.mean() for scores in values],
                    yerr=[scores.std() for scores in values], marker="o", capsize=4, label=label)
    ax.set(xlabel="Leaf Miner training source groups", ylabel="Validation score",
           title="Leaf Miner learning curve (MobileNetV2 + SVM)", ylim=(-0.03, 1.03))
    ax.set_xticks(counts)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Mean ± SD across subsets")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.MODELS_DIR / "mobilenetv2_cucumber.pth")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"AP's MobileNetV2 checkpoint is required: {args.checkpoint}")

    root = config.PROCESSED_DIR
    if os.name == "nt" and root.drive and not str(root).startswith("\\\\"):
        root = Path("\\\\?\\" + str(root))
    with (config.ROOT_DIR / "data" / "split_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    train_paths, train_labels, train_groups = load_split(root, manifest, "train")
    val_paths, val_labels, val_groups = load_split(root, manifest, "val")
    if set(train_groups) & set(val_groups):
        raise ValueError("Train and validation contain overlapping source groups")

    leaf_train = train_labels == LEAF_MINER
    leaf_val = val_labels == LEAF_MINER
    groups = np.unique(train_groups[leaf_train])
    if len(groups) < 4 or not leaf_val.any():
        raise ValueError("Learning curve requires at least four training groups and validation examples")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(config.CLASSES))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Extracting embeddings for {len(train_paths)} training images on {device} ...", flush=True)
    train_features, _ = embed(model, train_paths, device)
    print(f"Extracting embeddings for {len(val_paths)} validation images ...", flush=True)
    val_features, _ = embed(model, val_paths, device)

    orders = {seed: np.random.default_rng(seed).permutation(groups) for seed in SEEDS}
    results = []
    for fraction in FRACTIONS:
        count = max(1, int(fraction * len(groups) + 0.5))
        # The full training set is identical for every seed.
        subsets = [("all", groups)] if fraction == 1.0 else orders.items()
        for seed, order in subsets:
            selected = order[:count]
            keep = ~leaf_train | np.isin(train_groups, selected)
            svm = build_svm(SVM_C).fit(train_features[keep], train_labels[keep])
            predictions = svm.predict(val_features)
            correct = int(np.count_nonzero(predictions[leaf_val] == LEAF_MINER))
            row = {
                "target_fraction": fraction, "seed": seed, "svm_c": SVM_C,
                "leaf_miner_train_groups": count,
                "leaf_miner_train_images": int(np.count_nonzero(keep & leaf_train)),
                "leaf_miner_val_correct": correct, "leaf_miner_val_images": int(leaf_val.sum()),
                "val_leaf_miner_recall": correct / int(leaf_val.sum()),
                "val_macro_f1": macro_f1(val_labels, predictions),
            }
            results.append(row)
            print(f"{fraction:.0%} | seed {seed} | {count} groups, "
                  f"{row['leaf_miner_train_images']} crops | Leaf Miner {correct}/{leaf_val.sum()} | "
                  f"macro-F1 {row['val_macro_f1']:.4f}", flush=True)

    csv_path = config.RESULTS_DIR / "metrics" / "leaf_miner_learning_curve.csv"
    figure_path = config.RESULTS_DIR / "figures" / "leaf_miner_learning_curve.png"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plot_curve(results, figure_path)
    print(f"Saved {csv_path} and {figure_path}")


if __name__ == "__main__":
    main()
