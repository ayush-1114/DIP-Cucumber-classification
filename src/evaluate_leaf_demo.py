"""Evaluate the deployed predictor on the corrected manifest and verify saved predictions."""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import Counter

import cv2
import torch
from sklearn.metrics import classification_report, confusion_matrix

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corrected_data import OUT, photo
from src.leaf_demo import LABELS, analyze_rgb, load_demo_model


def main():
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dip_mobilenet", "dip_svm"))
    args = parser.parse_args()
    default = json.loads((OUT / "metrics.json").read_text())["demo"]
    winner = args.model or default
    model = load_demo_model(winner)
    with (OUT / "manifest.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    with (OUT / "test_predictions.csv").open(encoding="utf-8") as f:
        reference = {(r["path"], r["box"]): r[winner] for r in csv.DictReader(f)}
    if len(rows) != len(reference):
        raise ValueError("Test manifest and predictions differ")
    results = []
    for r in rows:
        _, stages, _, pred = analyze_rgb(photo(r), model)
        if pred != reference[r["path"], r["box"]]:
            raise ValueError(
                f"Deployment and training evaluation differ on {r['path']}"
            )
        results.append((r["label"], pred, stages["status"]))
    actual, pred, status = zip(*results)
    report = classification_report(
        actual, pred, labels=LABELS, digits=4, zero_division=0
    )
    report += "\nRows actual, columns predicted: " + ", ".join(LABELS) + "\n"
    report += (
        str(confusion_matrix(actual, pred, labels=LABELS))
        + "\n"
        + str(dict(Counter(status)))
    )
    full = [(a, b) for r, (a, b, _) in zip(rows, results) if not r["box"]]
    actual_full, pred_full = zip(*full)
    report += "\n\nFull-frame subset only (excludes extracted symptom regions)\n"
    report += classification_report(
        actual_full, pred_full, labels=LABELS, digits=4, zero_division=0
    )
    filename = (
        "deployed_test_report.txt"
        if winner == default
        else f"deployed_{winner}_report.txt"
    )
    (OUT / filename).write_text(report)
    print(winner, report, sep="\n")


if __name__ == "__main__":
    main()
