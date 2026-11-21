"""Classify one leaf photo and render the image-processing evidence."""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import models, transforms

from src.dip.pipeline import process_image, resize_with_padding, hsv_leaf_mask

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("Healthy Leaf", "Leaf Miner", "Red Mite", "Whitefly", "Yellow Mosaic")
PREPARE = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(filename):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(LABELS))
    model.load_state_dict(torch.load(ROOT / "models" / filename, map_location="cpu", weights_only=True))
    return model.eval()


@torch.inference_mode()
def predict(model, rgb):
    scores = model(PREPARE(rgb).unsqueeze(0))[0].softmax(0)
    return LABELS[int(scores.argmax())]


def analyze(photo, model):
    with Image.open(photo) as original:
        rgb = ImageOps.exif_transpose(original).convert("RGB")
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    stages = process_image(bgr)
    ok, encoded = cv2.imencode(".jpg", stages["final"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    prediction = predict(model, Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)))
    return bgr, stages, encoded, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/demo"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        bgr, stages, jpeg, label = analyze(args.photo, load_model("mobilenetv2_cucumber.pth"))
        directory = args.output / args.photo.stem
        directory.mkdir(parents=True, exist_ok=True)
        if args.photo.resolve() in {(directory / name).resolve() for name in ("processed.jpg", "stages.png")}:
            parser.error("Choose a different output directory to preserve the input.")
        jpeg.tofile(str(directory / "processed.jpg"))
        _, valid = resize_with_padding(bgr)
        seed = hsv_leaf_mask(stages["resized"], valid)
        items = [(bgr, "Original photograph"), (stages["resized"], "Aspect-preserving resize + padding"),
                 (seed, "HSV colour mask"), (stages["mask"], "Foreground mask"),
                 (stages["final"], "Masked CLAHE / classifier input"),
                 (stages["otsu"], "Otsu colour candidates"), (stages["edges"], "Canny interior edges")]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
        for ax in axes.flat:
            ax.axis("off")
        for ax, (img, title) in zip(axes.flat, items):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img,
                      cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=11)
        source_bytes = args.photo.stat().st_size
        metrics = (f"Prediction: {label}\nSource: {bgr.shape[1]} x {bgr.shape[0]} px\n"
                   f"Source: {source_bytes:,} bytes\n224 x 224 JPEG: {jpeg.size:,} bytes\n"
                   f"Storage reduction: {100*(1-jpeg.size/source_bytes):.1f}%")
        axes.flat[-1].text(0, 0.5, metrics, va="center", linespacing=1.6)
        fig.suptitle(f"Cucumber leaf analysis | {stages['status']}")
        fig.savefig(directory / "stages.png", dpi=140)
        plt.close(fig)
        print(metrics, f"\nSegmentation: {stages['status']}\nFigure: {directory / 'stages.png'}")
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Leaf demo failed: {exc}\n")


if __name__ == "__main__":
    main()
