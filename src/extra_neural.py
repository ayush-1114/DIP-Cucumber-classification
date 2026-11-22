"""Corrected-model stress tests, Grad-CAM checks and INT8 weight storage."""

import csv
import json
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn

from src.corrected_data import OUT, ROOT, NAMES, photo
from src.leaf_demo import load_model, PREPARE, prepare_rgb
from src.extra_corrected import EXTRA, save_csv, metrics


def inputs(rgb):
    _, stages, _, bgr = prepare_rgb(rgb)
    processed = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return {"raw": rgb, "dip": processed}, stages


def stress(rows, models):
    predictions = []
    for index, row in enumerate(rows):
        rgb = photo(row)
        original = np.asarray(rgb)
        variants = [
            ("clean", 0, original),
            ("brightness_60pct", 0, np.uint8(original * 0.6)),
        ]
        for seed in (42, 43, 44):
            rng = np.random.default_rng(seed + index * 100)
            noise = rng.normal(0, 25, original.shape).astype(np.float32)
            variants.append(
                (
                    "gaussian_std25_before_DIP",
                    seed,
                    np.clip(original.astype(np.float32) + noise, 0, 255).astype(
                        np.uint8
                    ),
                )
            )
        for condition, seed, image in variants:
            prepared, _ = inputs(Image.fromarray(image))
            for name, model in models.items():
                with torch.inference_mode():
                    scores = (
                        model(PREPARE(prepared[name]).unsqueeze(0))[0]
                        .softmax(0)
                        .numpy()
                    )
                predictions.append(
                    dict(
                        view=index,
                        path=row["path"],
                        group=row["group"],
                        actual=row["label"],
                        model=name,
                        condition=condition,
                        seed=seed,
                        prediction=NAMES[int(scores.argmax())],
                    )
                )
        if index % 20 == 0:
            print("Stress cases", index + 1, "/", len(rows), flush=True)
    save_csv("stress_predictions.csv", predictions)
    summary = []
    for name in models:
        for condition in ("clean", "brightness_60pct", "gaussian_std25_before_DIP"):
            for seed in (42, 43, 44) if condition.startswith("gaussian") else (0,):
                group = [
                    r
                    for r in predictions
                    if r["model"] == name
                    and r["condition"] == condition
                    and r["seed"] == seed
                ]
                actual = [NAMES.index(r["actual"]) for r in group]
                pred = [NAMES.index(r["prediction"]) for r in group]
                summary.append(
                    dict(
                        model=name,
                        condition=condition,
                        seed=seed,
                        **metrics(actual, pred),
                    )
                )
    save_csv("stress_summary.csv", summary)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    conditions = ("clean", "brightness_60pct", "gaussian_std25_before_DIP")
    for offset, name in ((-0.18, "raw"), (0.18, "dip")):
        means = [
            np.mean(
                [
                    r["accuracy"]
                    for r in summary
                    if r["model"] == name and r["condition"] == c
                ]
            )
            for c in conditions
        ]
        deviations = [
            np.std(
                [
                    r["accuracy"]
                    for r in summary
                    if r["model"] == name and r["condition"] == c
                ]
            )
            for c in conditions
        ]
        ax.bar(
            np.arange(3) + offset, means, 0.36, label=name, yerr=deviations, capsize=4
        )
    ax.set(
        xticks=range(3),
        xticklabels=["Clean", "Brightness 60%", "Gaussian noise, std 25/255"],
        ylabel="Test-view accuracy",
        ylim=(0, 1),
    )
    ax.legend()
    fig.savefig(EXTRA / "stress_comparison.png", dpi=150)
    plt.close(fig)


def attribution(rows, models):
    with (OUT / "test_predictions.csv").open() as f:
        reference = list(csv.DictReader(f))
    selections = []
    for label in NAMES:
        members = [i for i, r in enumerate(rows) if r["label"] == label]
        # First full-frame correct and first error per class, without inspecting heatmaps.
        correct = [i for i in members if reference[i]["dip_mobilenet"] == label]
        correct.sort(key=lambda i: bool(rows[i]["box"]))
        wrong = [i for i in members if reference[i]["dip_mobilenet"] != label]
        selections.extend(correct[:1] + wrong[:1])
    evidence = []
    for index in selections:
        row = rows[index]
        prepared, stages = inputs(photo(row))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
        for axis in axes.flat:
            axis.axis("off")
        for k, (name, model) in enumerate(models.items()):
            rgb = prepared[name]
            tensor = PREPARE(rgb).unsqueeze(0)
            with torch.no_grad():
                scores = model(tensor)[0].softmax(0)
            predicted = int(scores.argmax())
            actual = NAMES.index(row["label"])
            image = np.array(rgb.resize((224, 224)), np.float32) / 255
            axes[k, 0].imshow(image)
            axes[k, 0].set_title(
                f"{name}: predicted {NAMES[predicted]} ({scores[predicted]:.2f})"
            )
            with GradCAM(model=model, target_layers=[model.features[-1]]) as cam:
                for column, target in ((1, predicted), (2, actual)):
                    heat = cam(
                        input_tensor=tensor, targets=[ClassifierOutputTarget(target)]
                    )[0]
                    axes[k, column].imshow(show_cam_on_image(image, heat, use_rgb=True))
                    axes[k, column].set_title(f"Target {NAMES[target]}")
                    top = np.argsort(heat.ravel())[-int(0.2 * heat.size) :]
                    rng = np.random.default_rng(42 + index)
                    deletion = []
                    for pixels in (
                        top,
                        *[
                            rng.choice(heat.size, len(top), replace=False)
                            for _ in range(3)
                        ],
                    ):
                        changed = tensor.clone().reshape(1, 3, -1)
                        changed[:, :, pixels] = (
                            0  # ImageNet mean colour in normalized space.
                        )
                        with torch.no_grad():
                            probability = (
                                model(changed.reshape_as(tensor))[0]
                                .softmax(0)[target]
                                .item()
                            )
                        deletion.append(float(scores[target]) - probability)
                    evidence.append(
                        dict(
                            view=index,
                            path=row["path"],
                            actual=row["label"],
                            model=name,
                            predicted=NAMES[predicted],
                            target=NAMES[target],
                            cam_nonzero_fraction=float(np.mean(heat > 0)),
                            top20_deletion_score_drop=deletion[0],
                            random20_mean_drop=float(np.mean(deletion[1:])),
                            random20_std=float(np.std(deletion[1:])),
                        )
                    )
        fig.suptitle(
            f'Test view {index} | Actual {row["label"]} | {"symptom crop" if row["box"] else "full frame"}'
        )
        fig.savefig(EXTRA / f"gradcam_{index:03}.png", dpi=140)
        plt.close(fig)
    save_csv("gradcam_checks.csv", evidence)
    print("Attribution complete", len(selections), "preselected views", flush=True)


def quantization(rows, model):
    state = model.state_dict()
    quantized = {}
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            key = name + ".weight"
            value = state[key]
            scale = value.flatten(1).abs().amax(1) / 127
            scale = torch.where(scale > 0, scale, 1.0)
            shape = (-1,) + (1,) * (value.ndim - 1)
            quantized[key] = (
                (value / scale.reshape(shape)).round().clamp(-127, 127).to(torch.int8)
            )
            scales[key] = scale
    packed = dict(
        weights=quantized,
        scales=scales,
        other={k: v for k, v in state.items() if k not in quantized},
    )
    path = EXTRA / "dip_mobilenet_int8_storage.pth"
    torch.save(packed, path)
    packed = torch.load(path, weights_only=True)
    restored = dict(packed["other"])
    for key, value in packed["weights"].items():
        restored[key] = value.float() * packed["scales"][key].reshape(
            (-1,) + (1,) * (value.ndim - 1)
        )
    converted = load_model("corrected_dip_mobilenet.pth")
    converted.load_state_dict(restored)
    actual = []
    predictions = defaultdict(list)
    for row in rows:
        prepared, _ = inputs(photo(row))
        tensor = PREPARE(prepared["dip"]).unsqueeze(0)
        actual.append(NAMES.index(row["label"]))
        with torch.inference_mode():
            for name, network in (
                ("FP32", model),
                ("INT8 storage restored to FP32", converted),
            ):
                predictions[name].append(int(network(tensor).argmax(1)))
    baseline = (ROOT / "models/corrected_dip_mobilenet.pth").stat().st_size
    result = []
    for name, pred in predictions.items():
        size = baseline if name == "FP32" else path.stat().st_size
        result.append(
            dict(
                format=name,
                bytes=size,
                reduction_pct=100 * (1 - size / baseline),
                **metrics(actual, pred),
            )
        )
    save_csv("quantized_storage.csv", result)
    print("Weight storage experiment complete", flush=True)


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    EXTRA.mkdir(exist_ok=True)
    with (OUT / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    models = {
        "raw": load_model("corrected_raw_mobilenet.pth"),
        "dip": load_model("corrected_dip_mobilenet.pth"),
    }
    attribution(rows, models)
    quantization(rows, models["dip"])
    stress(rows, models)
    print("Corrected neural experiments complete", flush=True)


if __name__ == "__main__":
    main()
