"""Check the nine previously inspected web examples using their provisional filename labels."""

import csv
import cv2
import torch
from src.corrected_data import OUT, ROOT
from src.leaf_demo import analyze, load_demo_model


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    with (ROOT / "results/metrics/external_photo_diagnosis.csv").open() as f:
        cases = list(csv.DictReader(f))
    model = load_demo_model()
    rows = []
    for case in cases:
        path = ROOT / "demo_input" / case["filename"]
        _, stages, _, pred = analyze(path, model)
        rows.append(
            (
                path.name,
                case["provisional_expected_from_filename"],
                pred,
                max(stages["class_scores"].values()),
            )
        )
    with (OUT / "external_diagnostic.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("file", "provisional_expected", "prediction", "top_score"))
        w.writerows(rows)
    print(
        f"External diagnostic: {sum(expected==pred for _,expected,pred,_ in rows)}/{len(rows)}"
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
