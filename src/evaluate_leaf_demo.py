"""Benchmark the live DIP path on manifest-listed raw test images."""
import csv
from collections import Counter

import cv2
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.leaf_demo import ROOT, LABELS, analyze, load_model


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    model = load_model("mobilenetv2_cucumber.pth")
    records = []
    with (ROOT / "data/split_manifest.csv").open() as source:
        for row in csv.DictReader(source):
            if row["split"] != "test":
                continue
            photo = ROOT / "data/raw/test" / row["class"] / row["filename"]
            _, stages, _, label = analyze(photo, model)
            actual = "Healthy Leaf" if row["class"] == "Healthy" else row["class"].replace("_", " ")
            records.append((row["filename"], actual, label, stages["status"]))
    if not records:
        raise ValueError("The manifest has no test images")
    actual = [row[1] for row in records]
    predicted = [row[2] for row in records]
    destination = ROOT / "results/metrics"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "demo_predictions.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("filename", "actual", "prediction", "segmentation"))
        writer.writerows(records)
    report = classification_report(actual, predicted, labels=LABELS, digits=4, zero_division=0)
    report += "\nConfusion matrix (rows actual, columns predicted): " + ", ".join(LABELS)
    report += "\n" + str(confusion_matrix(actual, predicted, labels=LABELS)) + "\n"
    report += "\nSegmentation decisions\n"
    for status, count in Counter(row[3] for row in records).items():
        report += f"{status}: {count}\n"
    (destination / "demo_classification_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
