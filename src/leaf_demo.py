"""Classify one leaf photo and render the image-processing evidence."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import models, transforms

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dip.pipeline import process_image, resize_with_padding, hsv_leaf_mask
from src.corrected_data import NAMES as LABELS, ROOT

PREPARE = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model(filename):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(LABELS))
    model.load_state_dict(
        torch.load(ROOT / "models" / filename, map_location="cpu", weights_only=True)
    )
    return model.eval()


@torch.inference_mode()
def class_scores(model, rgb):
    return model(PREPARE(rgb).unsqueeze(0))[0].softmax(0).cpu().tolist()


def predict(model, rgb):
    scores = class_scores(model, rgb)
    return LABELS[max(range(len(scores)), key=scores.__getitem__)]


def prepare_rgb(rgb):
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    cv2.setRNGSeed(42)
    stages = process_image(bgr)
    ok, encoded = cv2.imencode(".jpg", stages["final"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return bgr, stages, encoded, decoded


def analyze_rgb(rgb, model):
    bgr, stages, encoded, decoded = prepare_rgb(rgb)
    if isinstance(model, torch.nn.Module):
        scores = class_scores(
            model, Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        )
    else:
        from src.classical_ml.features import extract_features

        margins = model.decision_function(extract_features(decoded).reshape(1, -1))[0]
        scores = torch.as_tensor(margins).softmax(0).tolist()
    prediction = LABELS[max(range(len(scores)), key=scores.__getitem__)]
    stages["class_scores"] = dict(zip(LABELS, scores))
    return bgr, stages, encoded, prediction


def load_demo_model(choice="auto"):
    if choice == "auto":
        metrics = ROOT / "results/corrected/metrics.json"
        choice = json.loads(metrics.read_text())["demo"]
    if choice == "dip_svm":
        import joblib

        return joblib.load(ROOT / "models/corrected_dip_svm.joblib")
    if choice != "dip_mobilenet":
        raise ValueError(f"Unknown model {choice}")
    return load_model("corrected_dip_mobilenet.pth")


def analyze(photo, model):
    with Image.open(photo) as original:
        return analyze_rgb(ImageOps.exif_transpose(original).convert("RGB"), model)


def run_demo(photo, model, output):
    bgr, stages, jpeg, label = analyze(photo, model)
    directory = output / photo.stem
    directory.mkdir(parents=True, exist_ok=True)
    if photo.resolve() in {
        (directory / name).resolve() for name in ("processed.jpg", "stages.png")
    }:
        raise ValueError("Choose a different output directory to preserve the input.")
    jpeg.tofile(str(directory / "processed.jpg"))
    _, valid = resize_with_padding(bgr)
    seed = hsv_leaf_mask(stages["resized"], valid)
    items = [
        (bgr, "Original photograph"),
        (stages["resized"], "Aspect-preserving resize + padding"),
        (seed, "HSV colour mask"),
        (stages["mask"], "Foreground mask"),
        (stages["final"], "Masked CLAHE / classifier input"),
        (stages["otsu"], "Otsu colour candidates"),
        (stages["edges"], "Canny interior edges"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img, title) in zip(axes.flat, items):
        ax.imshow(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img,
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        ax.set_title(title, fontsize=11)
    source_bytes = photo.stat().st_size
    metrics = (
        f"Prediction: {label}\nSource: {bgr.shape[1]} x {bgr.shape[0]} px\n"
        f"Source: {source_bytes:,} bytes\n224 x 224 JPEG: {jpeg.size:,} bytes\n"
        f"Storage reduction: {100*(1-jpeg.size/source_bytes):.1f}%"
    )
    score_text = "\n".join(
        f"{name:<14} {100*score:6.2f}%"
        for name, score in stages["class_scores"].items()
    )
    summary = f"{metrics}\n\nModel scores (softmax)\n{score_text}"
    axes.flat[-1].text(0, 0.5, summary, va="center", linespacing=1.4, fontsize=9)
    fig.suptitle(f"Cucumber leaf analysis | {stages['status']}")
    fig.savefig(directory / "stages.png", dpi=140)
    plt.close(fig)
    print(
        f"\nImage: {photo}\n{summary}",
        f"\nSegmentation: {stages['status']}\nFigure: {directory / 'stages.png'}",
    )
    return {
        "Image": photo.name,
        "Predicted class": label,
        **{f"{name} score (%)": round(100 * score, 4)
           for name, score in stages["class_scores"].items()},
        "Segmentation": stages["status"],
        "Retained area (%)": round(100 * cv2.countNonZero(stages["mask"]) / max(1, cv2.countNonZero(valid)), 2),
        "Error": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "photo",
        nargs="?",
        type=Path,
        default=ROOT / "demo_input",
        help="Photo or folder of unlabelled images (default: demo_input)",
    )
    parser.add_argument(
        "--model", choices=["auto", "dip_mobilenet", "dip_svm"], default="auto"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/demo")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.photo.is_dir():
            photos = sorted(
                p
                for p in args.photo.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
            if not photos:
                parser.error(
                    f"Place a JPEG or PNG photo in {args.photo}, then run again."
                )
        else:
            photos = [args.photo]
        if len({p.stem.casefold() for p in photos}) != len(photos):
            parser.error(
                "Give each photo a unique filename stem to keep its outputs separate."
            )
        choice = args.model
        if choice == "auto":
            choice = json.loads((ROOT / "results/corrected/metrics.json").read_text())["demo"]
        model = load_demo_model(choice)
        failures = []
        rows = []
        for photo in photos:
            try:
                row = run_demo(photo, model, args.output)
            except (OSError, ValueError, cv2.error) as exc:
                failures.append(photo)
                print(f"Could not process {photo}: {exc}")
                row = {"Image": photo.name, "Error": str(exc)}
            rows.append({**row, "Model": choice})
        args.output.mkdir(parents=True, exist_ok=True)
        csv_path = args.output / "predictions.csv"
        fields = ["Image", "Predicted class", *[f"{name} score (%)" for name in LABELS],
                  "Model", "Segmentation", "Retained area (%)", "Error"]
        try:
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        except PermissionError:
            parser.exit(1, f"Close {csv_path} in Excel, then rerun to refresh the CSV.\n")
        print(f"\nResults CSV: {csv_path}")
        if failures:
            parser.exit(1, f"{len(failures)} image(s) failed. See messages above.\n")
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Leaf demo failed: {exc}\n")


if __name__ == "__main__":
    main()
