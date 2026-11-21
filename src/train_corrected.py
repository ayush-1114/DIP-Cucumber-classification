"""Fit new models from ImageNet and source-grouped data; run python -m src.train_corrected."""

import copy
import csv
import json
import sys
from collections import Counter
from importlib.metadata import version

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import models

from src.classical_ml.features import extract_features
from src.corrected_data import OUT, ROOT, NAMES, build, disk, photo
from src.leaf_demo import PREPARE, prepare_rgb


def features(rows, backbone):
    raw, dip, hand, stored = [], [], [], []
    for start in range(0, len(rows), 12):
        batch = rows[start : start + 12]
        raw_batch, dip_batch = [], []
        for row in batch:
            rgb = photo(row)
            _, _, encoded, bgr = prepare_rgb(rgb)
            stored.append(len(encoded))
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            raw_batch.append(PREPARE(rgb))
            dip_batch.append(PREPARE(image))
            hand.append(extract_features(bgr))
        with torch.no_grad():
            for images, features_out in [(raw_batch, raw), (dip_batch, dip)]:
                x = torch.stack(images)
                features_out.append(backbone.features(x).mean((2, 3)))
        if start % 120 == 0:
            print(f"Prepared {start+len(batch)}/{len(rows)} views", flush=True)
    return torch.cat(raw), torch.cat(dip), np.stack(hand), stored


def fit_head(x, y, parts, backbone, weights):
    torch.manual_seed(42)
    head = torch.nn.Linear(x.shape[1], len(NAMES))
    opt = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.01)
    tr = np.where(parts == "train")[0]
    va = np.where(parts == "val")[0]
    best = -1
    for epoch in range(40):
        for b in torch.tensor(tr)[torch.randperm(len(tr))].split(32):
            loss = torch.nn.functional.cross_entropy(head(x[b]), y[b], reduction="none")
            opt.zero_grad()
            (loss * weights[b]).mean().backward()
            opt.step()
        with torch.no_grad():
            pred = head(x[va]).argmax(1)
            score = f1_score(y[va], pred, average="macro")
        if score > best:
            best, chosen, state = score, epoch + 1, copy.deepcopy(head.state_dict())
    head.load_state_dict(state)
    backbone.classifier[1] = head
    with torch.no_grad():
        return head(x).argmax(1).numpy(), float(best), chosen


def main():
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    rows = build()
    packages = (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "scikit-learn",
        "scikit-image",
        "scipy",
        "pillow",
        "matplotlib",
        "joblib",
        "requests",
    )
    (OUT / "runtime_versions.json").write_text(
        json.dumps(
            {"python": sys.version, "packages": {p: version(p) for p in packages}},
            indent=2,
        )
    )
    (ROOT / "models").mkdir(exist_ok=True)
    print("Source split", Counter(r["split"] for r in rows), flush=True)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()
    xraw, xdip, features_hand, sizes = features(rows, model)
    parts = np.array([r["split"] for r in rows])
    y = torch.tensor([NAMES.index(r["label"]) for r in rows])
    # Equal total weight per class and equal weight per source photo group within class.
    train = [r for r in rows if r["split"] == "train"]
    counts = Counter(r["group"] for r in train)
    groups = {
        label: {r["group"] for r in train if r["label"] == label} for label in NAMES
    }
    weights = torch.tensor(
        [
            (
                len(train) / (5 * len(groups[r["label"]]) * counts[r["group"]])
                if r["split"] == "train"
                else 0
            )
            for r in rows
        ]
    )
    predictions = {}
    validation = {}
    epochs = {}
    for name, x in (("raw_mobilenet", xraw), ("dip_mobilenet", xdip)):
        predictions[name], validation[name], epochs[name] = fit_head(
            x, y, parts, model, weights
        )
        torch.save(model.state_dict(), ROOT / "models" / f"corrected_{name}.pth")
    tr = parts == "train"
    va = parts == "val"
    best = -1
    for c in (0.1, 1, 10):
        svm = make_pipeline(
            StandardScaler(), SVC(C=c, decision_function_shape="ovr", break_ties=True)
        )
        svm.fit(features_hand[tr], y[tr], svc__sample_weight=weights[tr].numpy())
        score = f1_score(y[va], svm.predict(features_hand[va]), average="macro")
        if score > best:
            best, chosen, selected = score, c, svm
    joblib.dump(selected, ROOT / "models/corrected_dip_svm.joblib", compress=3)
    predictions["dip_svm"] = selected.predict(features_hand)
    validation["dip_svm"] = float(best)
    # For the DIP demonstration, choose between two models that consume the actual DIP output.
    winner = max(("dip_mobilenet", "dip_svm"), key=validation.get)
    summary = dict(
        validation_macro_f1=validation,
        selected_epoch=epochs,
        svm_C=chosen,
        demo=winner,
        split_counts=dict(Counter(parts)),
        source_groups={
            p: len({r["group"] for r in rows if r["split"] == p}) for p in set(parts)
        },
        test={},
        storage={},
    )
    test = parts == "test"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, (name, pred) in zip(axes, predictions.items()):
        report = classification_report(
            y[test],
            pred[test],
            labels=list(range(5)),
            target_names=NAMES,
            output_dict=True,
            zero_division=0,
        )
        summary["test"][name] = report
        ConfusionMatrixDisplay.from_predictions(
            y[test],
            pred[test],
            labels=list(range(5)),
            display_labels=NAMES,
            ax=ax,
            colorbar=False,
            xticks_rotation=45,
            cmap="Blues",
        )
        ax.set_title(name.replace("_", " "))
    fig.savefig(OUT / "confusion_matrices.png", dpi=140)
    plt.close(fig)
    # Count full images only so raw bytes and processed bytes refer to identical content.
    full = [i for i, r in enumerate(rows) if not r["box"]]
    original = sum(disk(rows[i]["path"]).stat().st_size for i in full)
    processed = sum(sizes[i] for i in full)
    summary["storage"] = dict(
        full_images=len(full),
        original_bytes=original,
        processed_bytes=processed,
        reduction_pct=100 * (1 - processed / original),
    )
    with (OUT / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "box", "group", "actual", *predictions])
        for i in np.flatnonzero(test):
            w.writerow(
                [
                    rows[i]["path"],
                    rows[i]["box"],
                    rows[i]["group"],
                    rows[i]["label"],
                    *[NAMES[p[i]] for p in predictions.values()],
                ]
            )
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("Validation", validation, "DIP demo", winner, flush=True)
    print(
        "Test",
        {
            n: (v["accuracy"], v["macro avg"]["f1-score"])
            for n, v in summary["test"].items()
        },
    )
    print("Storage", summary["storage"])


if __name__ == "__main__":
    main()
