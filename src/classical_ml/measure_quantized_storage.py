"""Measure MobileNetV2 checkpoint storage with per-channel INT8 weights."""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

from src import config
from src.classical_ml.leaf_miner_learning_curve import load_split
from src.classical_ml.mobilenet_svm import EmbeddingImages
from src.classical_ml.train import macro_f1


def quantize_weights(model, state):
    int8_weights = {}
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            key = f"{name}.weight"
            weights = state[key]
            shape = (-1,) + (1,) * (weights.ndim - 1)
            scale = weights.flatten(1).abs().amax(dim=1) / 127
            scale = torch.where(scale > 0, scale, 1.0)
            int8_weights[key] = (weights / scale.reshape(shape)).round().clamp(-127, 127).to(torch.int8)
            scales[key] = scale
    return {
        "int8_weights": int8_weights,
        "scales": scales,
        "other_state": {key: value for key, value in state.items() if key not in int8_weights},
    }


def restore_state(checkpoint):
    state = dict(checkpoint["other_state"])
    for key, weights in checkpoint["int8_weights"].items():
        shape = (-1,) + (1,) * (weights.ndim - 1)
        state[key] = weights.float() * checkpoint["scales"][key].reshape(shape)
    return state


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    source = config.MODELS_DIR / "mobilenetv2_cucumber.pth"
    target = source.with_name("mobilenetv2_cucumber_int8.pth")
    root = config.PROCESSED_DIR
    if os.name == "nt" and root.drive and not str(root).startswith("\\\\"):
        root = Path("\\\\?\\" + str(root))
    with (config.ROOT_DIR / "data" / "split_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        paths, labels, _ = load_split(root, list(csv.DictReader(handle)), "test")
    test = DataLoader(EmbeddingImages(paths), batch_size=16)

    model = mobilenet_v2(weights=None, num_classes=len(config.CLASSES)).eval()
    original = torch.load(source, map_location="cpu", weights_only=True)
    torch.save(quantize_weights(model, original), target)
    stored = torch.load(target, map_location="cpu", weights_only=True)
    if not stored["int8_weights"] or any(
        tensor.dtype != torch.int8 for tensor in stored["int8_weights"].values()
    ):
        raise ValueError("Expected saved INT8 weights")

    ap_to_project = np.array([config.CLASS_TO_IDX[name] for name in sorted(config.CLASSES)])
    leaf = labels == config.CLASS_TO_IDX["Leaf_Miner"]
    original_bytes = source.stat().st_size
    rows = []
    with torch.inference_mode():
        for name, state, checkpoint in (("FP32", original, source),
                                         ("INT8 weight storage", restore_state(stored), target)):
            print(f"Evaluating {name} on {len(paths)} test crops", flush=True)
            model.load_state_dict(state)
            indices = np.concatenate([model(images).argmax(1).numpy() for images in test])
            predictions = ap_to_project[indices]
            size = checkpoint.stat().st_size
            rows.append({
                "precision": name, "checkpoint": checkpoint.name, "bytes": size,
                "mb": size / 1_000_000, "reduction_pct": 100 * (1 - size / original_bytes),
                "test_correct": int(np.count_nonzero(predictions == labels)),
                "test_images": len(labels), "test_accuracy": accuracy_score(labels, predictions),
                "test_macro_f1": macro_f1(labels, predictions),
                "leaf_miner_correct": int(np.count_nonzero(predictions[leaf] == labels[leaf])),
                "leaf_miner_total": int(leaf.sum()),
            })
            print(rows[-1], flush=True)

    output = config.RESULTS_DIR / "storage_reduction" / "mobilenetv2_quantization.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {target} and {output}")


if __name__ == "__main__":
    main()
