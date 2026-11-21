"""Compare storage after each DIP operation at fixed JPEG quality on the test split."""

import csv
import cv2
import numpy as np
from PIL import Image, ImageOps
from src.dip.pipeline import resize_with_padding, apply_clahe, apply_grabcut
from src.corrected_data import OUT, disk


def main():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    totals = dict.fromkeys(
        (
            "Original file",
            "Resized JPEG",
            "Resized + CLAHE JPEG",
            "Resized + CLAHE + foreground mask JPEG",
        ),
        0,
    )
    count = 0
    with (OUT / "manifest.csv").open() as source:
        for row in csv.DictReader(source):
            if row["split"] != "test" or row["box"]:
                continue
            path = disk(row["path"])
            with Image.open(path) as image:
                bgr = cv2.cvtColor(
                    np.asarray(ImageOps.exif_transpose(image).convert("RGB")),
                    cv2.COLOR_RGB2BGR,
                )
            resized, valid = resize_with_padding(bgr)
            enhanced = apply_clahe(resized)
            cv2.setRNGSeed(42)
            mask, _ = apply_grabcut(resized, valid)
            final = cv2.bitwise_and(enhanced, enhanced, mask=mask)
            totals["Original file"] += path.stat().st_size
            for stage, pixels in zip(list(totals)[1:], (resized, enhanced, final)):
                ok, data = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise ValueError(f"Encoding failed for {path}")
                totals[stage] += data.size
            count += 1
    with (OUT / "dip_stage_storage.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("stage", "test_images", "bytes", "reduction_vs_original_pct"))
        for stage, size in totals.items():
            writer.writerow(
                (
                    stage,
                    count,
                    size,
                    round(100 * (1 - size / totals["Original file"]), 2),
                )
            )


if __name__ == "__main__":
    main()
