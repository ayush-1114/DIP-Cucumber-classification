"""Compare segmentation ablations using saved weights without fitting to demo photos."""

import csv

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from src.corrected_data import ROOT, NAMES
from src.dip.pipeline import resize_with_padding, hsv_leaf_mask, process_image, apply_clahe
from src.leaf_demo import load_demo_model, class_scores


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = load_demo_model()
    rows = []
    for path in sorted((ROOT / "demo_input").glob("*.jpg")):
        with Image.open(path) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        resized, valid = resize_with_padding(bgr)
        prior = hsv_leaf_mask(resized, valid)
        cv2.setRNGSeed(42)
        stages = process_image(bgr)
        gc = np.full(valid.shape, cv2.GC_BGD, np.uint8)
        gc[valid > 0] = cv2.GC_PR_BGD
        # A fixed central rectangle tests whether a spatial prior can remove green background.
        h, w = bgr.shape[:2]
        scale = 224 / max(h, w)
        rh, rw = round(h * scale), round(w * scale)
        y, x = (224 - rh) // 2, (224 - rw) // 2
        gc[y + rh // 8:y + 7 * rh // 8, x + rw // 5:x + 4 * rw // 5] = cv2.GC_PR_FGD
        cv2.setRNGSeed(42)
        cv2.grabCut(resized, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
        rectangle = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        enhanced = apply_clahe(resized)
        masks = {"deployed": stages["mask"], "no_mask": valid, "hsv_mask": prior, "central_grabcut": rectangle}
        for mode, mask in masks.items():
            image = cv2.bitwise_and(enhanced, enhanced, mask=mask)
            _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            scores = class_scores(model, Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)))
            rows.append({"Image": path.name, "Variant": mode, "Prediction": NAMES[int(np.argmax(scores))],
                         "HSV coverage (%)": round(100 * cv2.countNonZero(prior) / cv2.countNonZero(valid), 2),
                         "Kept area (%)": round(100 * cv2.countNonZero(mask) / cv2.countNonZero(valid), 2),
                         **{f"{n} score (%)": round(100*s, 3) for n,s in zip(NAMES,scores)}})
    output = ROOT / "results/corrected/segmentation_diagnostic.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for r in rows:
        print(r["Image"], r["Variant"], r["Prediction"], r["HSV coverage (%)"], r["Kept area (%)"])


if __name__ == "__main__":
    main()
