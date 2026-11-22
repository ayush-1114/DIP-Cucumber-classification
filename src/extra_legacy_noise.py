"""Reproduce the historical whole-tensor Gaussian stress condition with fixed paired seeds."""

import csv
import torch
from src.corrected_data import OUT, NAMES, photo
from src.leaf_demo import PREPARE, load_model
from src.extra_neural import inputs
from src.extra_corrected import save_csv, metrics
import cv2


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with (OUT / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    models = {
        "raw": load_model("corrected_raw_mobilenet.pth"),
        "dip": load_model("corrected_dip_mobilenet.pth"),
    }
    output = []
    for index, row in enumerate(rows):
        prepared, _ = inputs(photo(row))
        for seed in (42, 43, 44):
            generator = torch.Generator().manual_seed(seed + 100 * index)
            noise = torch.randn((1, 3, 224, 224), generator=generator) * 0.8
            for name, model in models.items():
                tensor = PREPARE(prepared[name]).unsqueeze(0) + noise
                with torch.inference_mode():
                    pred = int(model(tensor).argmax(1))
                output.append(
                    dict(
                        view=index,
                        actual=row["label"],
                        model=name,
                        seed=seed,
                        prediction=NAMES[pred],
                    )
                )
    save_csv("legacy_tensor_noise_predictions.csv", output)
    summary = []
    for name in models:
        for seed in (42, 43, 44):
            group = [r for r in output if r["model"] == name and r["seed"] == seed]
            summary.append(
                dict(
                    model=name,
                    seed=seed,
                    **metrics(
                        [NAMES.index(r["actual"]) for r in group],
                        [NAMES.index(r["prediction"]) for r in group],
                    )
                )
            )
    save_csv("legacy_tensor_noise_summary.csv", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
