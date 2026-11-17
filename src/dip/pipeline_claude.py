"""DIP preprocessing pipeline, Iteration 3 (suggested fix for the Iteration 2 failures).

Order of steps:
  1. Resize with padding to 224x224
  2. HSV plant mask                (seed for GrabCut, replaces the Otsu seed)
  3. GrabCut, only when the image actually has background
  4. CLAHE on the L channel
  5. Otsu inside the leaf only     (lesions vs healthy tissue)
  6. Canny inside the leaf only

Try it on one image (saves the stages side by side):

    python pipeline.py path/to/image.jpg stages.jpg
    python pipeline.py path/to/crop.jpg stages.jpg --no-segment

See README.md for the reasoning and the known weak spots.
"""

import sys

import cv2
import numpy as np

IMG_SIZE = 224
MIN_FG, MAX_FG = 0.10, 0.85   # outside this range the colour mask is not trusted


def resize_with_padding(image, size=IMG_SIZE):
    """Resize keeping aspect ratio, pad with black. Also return a mask of the real pixels."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    valid = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    valid[y0:y0 + nh, x0:x0 + nw] = 255
    return canvas, valid


def fill_holes(mask):
    """Fill everything inside the outer contours (keeps a white bug inside a green leaf)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    min_area = 0.02 * mask.size
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    cv2.drawContours(filled, big, -1, 255, cv2.FILLED)
    return filled


def hsv_leaf_mask(image, valid):
    """Rough 'plant material' mask: coloured pixels. White/grey backdrops have low saturation."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # plant colours only: hue from brown/yellow to green, reasonably saturated, not near-black
    plant = (h >= 8) & (h <= 100) & (s > 35) & (v > 40)
    mask = np.where(plant, 255, 0).astype(np.uint8)
    mask = cv2.bitwise_and(mask, valid)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return fill_holes(mask)


def apply_grabcut(image, valid):
    """Return (foreground mask, status). Never returns an empty or near-empty mask."""
    prior = hsv_leaf_mask(image, valid)
    frac = cv2.countNonZero(prior) / max(1, cv2.countNonZero(valid))
    if frac > MAX_FG:
        return valid.copy(), "skipped: tight crop, no background"
    if frac < MIN_FG:
        return valid.copy(), "skipped: no reliable colour mask"

    gc = np.full(image.shape[:2], cv2.GC_PR_BGD, np.uint8)
    gc[prior == 255] = cv2.GC_PR_FGD
    gc[cv2.erode(prior, np.ones((15, 15), np.uint8)) == 255] = cv2.GC_FGD
    gc[cv2.dilate(prior, np.ones((25, 25), np.uint8)) == 0] = cv2.GC_BGD
    gc[valid == 0] = cv2.GC_BGD                      # padding is certainly background
    if not ((gc == cv2.GC_FGD).any() and (gc == cv2.GC_BGD).any()):
        return prior, "used HSV mask: no sure seeds for GrabCut"

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = fill_holes(fg)
    if cv2.countNonZero(fg) < 0.5 * cv2.countNonZero(prior):
        return prior, "used HSV mask: GrabCut removed too much"
    return fg, "grabcut"


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_otsu_lesions(image, fg):
    """Otsu INSIDE the leaf only, on the LAB a* channel (green = low, yellow/brown = high)."""
    a = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 1], (5, 5), 0)
    pixels = a[fg == 255]
    out = np.zeros_like(fg)
    if pixels.size < 50 or pixels.std() < 2:          # nothing to separate
        return out
    t, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out[(a > t) & (fg == 255)] = 255
    return out


def apply_canny(image, fg):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    inner = cv2.erode(fg, np.ones((5, 5), np.uint8))  # drop the artificial mask outline
    return cv2.bitwise_and(edges, inner)


def process_image(bgr, segment=True):
    """Run the full pipeline on one BGR image.

    segment=False skips GrabCut and keeps the whole image. Use it for images that are
    already tight crops with no background (the Roboflow Whitefly / Leaf_Miner crops).

    Returns a dict with every stage, for the Feature Evolution Plots:
      resized  - 224x224 padded original
      mask     - foreground (leaf) mask, 0/255
      final    - CLAHE-enhanced image with the background removed  <- model input
      otsu     - lesion mask from Otsu inside the leaf, 0/255
      edges    - Canny edges inside the leaf, 0/255
      status   - what the segmentation step did (for logging)
    """
    resized, valid = resize_with_padding(bgr)
    if segment:
        fg, status = apply_grabcut(resized, valid)       # segment on the ORIGINAL colours
    else:
        fg, status = valid.copy(), "skipped: segment=False"
    enhanced = apply_clahe(resized)                       # then enhance
    final = cv2.bitwise_and(enhanced, enhanced, mask=fg)
    lesions = apply_otsu_lesions(enhanced, fg)
    edges = apply_canny(enhanced, fg)
    return {"resized": resized, "mask": fg, "final": final,
            "otsu": lesions, "edges": edges, "status": status}


def stage_strip(result):
    """All stages side by side: resized | mask | final | otsu | edges."""
    grey = lambda m: cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return np.hstack([result["resized"], grey(result["mask"]), result["final"],
                      grey(result["otsu"]), grey(result["edges"])])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-segment"]
    if len(args) != 2:
        raise SystemExit(__doc__)
    # np.fromfile + imdecode instead of cv2.imread: imread fails on some Windows paths
    image = cv2.imdecode(np.fromfile(args[0], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args[0]}")
    result = process_image(image, segment="--no-segment" not in sys.argv)
    cv2.imwrite(args[1], stage_strip(result))
    print(f"Segmentation: {result['status']}")
    print(f"Saved {args[1]} (resized | mask | final | otsu | edges)")
