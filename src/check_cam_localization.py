"""Evaluate Grad-CAM localization against publisher Leaf Miner damage rectangles."""

import csv
import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.corrected_data import OUT, disk
from src.dip.pipeline import resize_with_padding
from src.leaf_demo import load_model, PREPARE
from src.extra_corrected import save_csv
from src.extra_neural import inputs
from src.corrected_data import photo


def annotated_mask(row):
    path = disk(row["path"])
    with Image.open(path) as image:
        assert (
            image.getexif().get(274, 1) == 1
        ), "Annotation orientation must match image"
        width, height = image.size
    mask = np.zeros((height, width), np.uint8)
    labels = disk("external_data/wmc4/labels/" + path.stem + ".txt")
    for line in labels.read_text().splitlines():
        values = line.split()
        if len(values) != 5 or values[0] != "0":
            continue
        x, y, w, h = map(float, values[1:])
        mask[
            max(0, int((y - h / 2) * height)) : min(
                height, int((y + h / 2) * height) + 1
            ),
            max(0, int((x - w / 2) * width)) : min(width, int((x + w / 2) * width) + 1),
        ] = 255
    if row["box"]:
        x, y, w, h = map(float, row["box"].split())
        mask = mask[
            max(0, int((y - h * 0.8) * height)) : min(
                height, int((y + h * 0.8) * height) + 1
            ),
            max(0, int((x - w * 0.8) * width)) : min(
                width, int((x + w * 0.8) * width) + 1
            ),
        ]
    return mask


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with (OUT / "manifest.csv").open() as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["split"] == "test" and r["source"] == "wmc_miner"
        ]
    models = {
        "raw": load_model("corrected_raw_mobilenet.pth"),
        "dip": load_model("corrected_dip_mobilenet.pth"),
    }
    records = []
    for row in rows:
        original = annotated_mask(row)
        prepared, _ = inputs(photo(row))
        for name, model in models.items():
            mask = (
                cv2.resize(original, (224, 224), interpolation=cv2.INTER_NEAREST)
                if name == "raw"
                else resize_with_padding(cv2.cvtColor(original, cv2.COLOR_GRAY2BGR))[0][
                    :, :, 0
                ]
            )
            inside = mask > 127
            tensor = PREPARE(prepared[name]).unsqueeze(0)
            with GradCAM(model=model, target_layers=[model.features[-1]]) as cam:
                heat = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(1)])[0]
            total = float(heat.sum())
            energy = float(heat[inside].sum()) / total if total > 0 else float("nan")
            area = float(inside.mean())
            peak = np.unravel_index(heat.argmax(), heat.shape)
            records.append(
                dict(
                    path=row["path"],
                    group=row["group"],
                    box=row["box"],
                    view_type="crop" if row["box"] else "full_frame",
                    model=name,
                    annotated_area_fraction=area,
                    cam_energy_in_damage_boxes=energy,
                    enrichment_vs_uniform=energy / area if area else float("nan"),
                    peak_in_damage_box=bool(inside[peak]) if total > 0 else False,
                )
            )
    save_csv("leaf_miner_localization.csv", records)
    print("Localization evaluated", len(records), "model/view pairs", flush=True)


if __name__ == "__main__":
    main()
