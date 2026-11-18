"""Plot t-SNE projections of HOG and MobileNetV2 features on the shared test set."""

import argparse
import csv
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from torch import nn
from torchvision import models

from src import config
from src.classical_ml.features import hog_features, load_image
from src.classical_ml.leaf_miner_learning_curve import load_split
from src.classical_ml.mobilenet_svm import embed


def project(features):
    with threadpool_limits(limits=2):
        reduced = PCA(n_components=30, random_state=config.SEED).fit_transform(features)
        return TSNE(n_components=2, perplexity=20, init="pca",
                    learning_rate="auto", random_state=config.SEED).fit_transform(reduced)


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
        paths, labels, _ = load_split(root, list(csv.DictReader(handle)), "test")

    print(f"Extracting HOG for {len(paths)} test images", flush=True)
    hog = np.stack([hog_features(cv2.cvtColor(load_image(path), cv2.COLOR_BGR2GRAY))
                    for path in paths])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(config.CLASSES))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    print(f"Extracting MobileNetV2 embeddings on {device}", flush=True)
    embeddings, _ = embed(model, paths, device)

    styles = {
        "Healthy": ("#00866A", "o"),
        "Leaf_Miner": ("#C04500", "D"),
        "Red_Mite": ("#8B529B", "o"),
        "Whitefly": ("#0072B2", "^"),
        "Yellow_Mosaic": ("#C49200", "o"),
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.7), layout="constrained")
    fig.suptitle("Cucumber disease classes in two feature spaces", fontsize=17, fontweight="bold")
    handles = {}
    for ax, title, features in zip(
        axes, ("HOG only", "MobileNetV2 embeddings"), (hog, embeddings)
    ):
        coordinates = project(features)
        ax.set(title=title, xticks=[], yticks=[])
        ax.set_facecolor("#F5F7FA")
        for spine in ax.spines.values():
            spine.set_color("#D0D8E1")
        for name in ("Healthy", "Red_Mite", "Yellow_Mosaic", "Whitefly", "Leaf_Miner"):
            points = coordinates[labels == config.CLASS_TO_IDX[name]]
            color, marker = styles[name]
            highlight = name in ("Leaf_Miner", "Whitefly")
            handles[name] = ax.scatter(
                points[:, 0], points[:, 1], c=color, marker=marker,
                s=49 if highlight else 32, edgecolors="white", linewidths=0.55,
                alpha=0.96 if highlight else 0.74, zorder=3 if highlight else 2,
                label=f"{name.replace('_', ' ')} ({len(points)})",
            )
    fig.legend(handles.values(), [handle.get_label() for handle in handles.values()],
               loc="outside lower center", ncols=5, frameon=False)
    output = config.RESULTS_DIR / "figures" / "hog_mobilenet_tsne.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
