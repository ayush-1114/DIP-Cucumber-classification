"""Evaluate a saved demo model on independently labelled, full-frame photographs."""

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import torch
from sklearn.metrics import classification_report

from src.corrected_data import NAMES, OUT
from src.leaf_demo import analyze, load_demo_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        type=Path,
        help="CSV with path,label,source. Paths are relative to the CSV.",
    )
    parser.add_argument(
        "--model", choices=("auto", "dip_mobilenet", "dip_svm"), default="auto"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New CSV for predictions and five scores",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output filename to keep previous runs.")
    with args.cases.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"path", "label", "source"} <= set(reader.fieldnames or []):
            parser.error("Expected CSV headers path,label,source.")
        cases = list(reader)
    if not cases:
        parser.error("The case list is empty.")
    known = set((OUT / "source_sha256.txt").read_text().splitlines())
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    model = load_demo_model(args.model)
    predictions = []
    for case in cases:
        path = args.cases.parent / case["path"]
        if case["label"] not in NAMES or not case["source"].strip():
            parser.error(f"Missing source or unknown class for {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in known:
            parser.error(f"Duplicate benchmark or evaluation image: {path}")
        known.add(digest)
        _, stages, _, prediction = analyze(path, model)
        predictions.append(
            {
                **{k: case[k] for k in ("path", "label", "source")},
                "prediction": prediction,
                **stages["class_scores"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(
        classification_report(
            [r["label"] for r in predictions],
            [r["prediction"] for r in predictions],
            labels=NAMES,
            zero_division=0,
            digits=4,
        )
    )
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
