import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PLOTS_DIR = ROOT_DIR / "results" / "figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CLAUDE'S ITERATION 3 FUNCTIONS
# ==========================================
IMG_SIZE = 224
MIN_FG, MAX_FG = 0.10, 0.85

def resize_with_padding(image, size=IMG_SIZE):
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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    min_area = 0.02 * mask.size
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    cv2.drawContours(filled, big, -1, 255, cv2.FILLED)
    return filled

def hsv_leaf_mask(image, valid):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    plant = (h >= 8) & (h <= 100) & (s > 35) & (v > 40)
    mask = np.where(plant, 255, 0).astype(np.uint8)
    mask = cv2.bitwise_and(mask, valid)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return fill_holes(mask)

def apply_grabcut(image, valid):
    prior = hsv_leaf_mask(image, valid)
    frac = cv2.countNonZero(prior) / max(1, cv2.countNonZero(valid))
    if frac > MAX_FG:
        return valid.copy(), "full frame retained: HSV coverage >85%; background unresolved"
    if frac < MIN_FG:
        return valid.copy(), "full frame retained: HSV coverage <10%; foreground unresolved"

    gc = np.full(image.shape[:2], cv2.GC_PR_BGD, np.uint8)
    gc[prior == 255] = cv2.GC_PR_FGD
    gc[cv2.erode(prior, np.ones((15, 15), np.uint8)) == 255] = cv2.GC_FGD
    gc[cv2.dilate(prior, np.ones((25, 25), np.uint8)) == 0] = cv2.GC_BGD
    gc[valid == 0] = cv2.GC_BGD
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
    a = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 1], (5, 5), 0)
    pixels = a[fg == 255]
    out = np.zeros_like(fg)
    if pixels.size < 50 or pixels.std() < 2:
        return out
    t, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out[(a > t) & (fg == 255)] = 255
    return out

def apply_canny(image, fg):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    inner = cv2.erode(fg, np.ones((5, 5), np.uint8))
    return cv2.bitwise_and(edges, inner)

def process_image(bgr, segment=True):
    resized, valid = resize_with_padding(bgr)
    if segment:
        fg, status = apply_grabcut(resized, valid)
    else:
        fg, status = valid.copy(), "skipped: segment=False"
    enhanced = apply_clahe(resized)
    final = cv2.bitwise_and(enhanced, enhanced, mask=fg)
    lesions = apply_otsu_lesions(enhanced, fg)
    edges = apply_canny(enhanced, fg)
    return {"resized": resized, "mask": fg, "final": final,
            "otsu": lesions, "edges": edges, "status": status}

# ==========================================
# BATCH PROCESSING AND PLOTTING
# ==========================================
def plot_feature_evolution(result, save_path, class_name):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    # Order: Original -> GrabCut(Mask applied) -> CLAHE -> Otsu -> Edges
    keys = [('resized', 'Original'), ('mask', 'Mask'), ('final', 'Final (CLAHE)'), ('otsu', 'Otsu'), ('edges', 'Edges')]
    
    for ax, (key, title) in zip(axes, keys):
        img = result[key]
        if len(img.shape) == 2: # Grayscale masks
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
            
    plt.suptitle(f"Feature Evolution: {class_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_pipeline():
    # Process train, val, and test splits
    for split in ["train", "val", "test"]:
        split_dir = RAW_DIR / split
        if not split_dir.exists():
            continue
            
        classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
        for cls in classes:
            raw_class_dir = split_dir / cls
            processed_class_dir = PROCESSED_DIR / split / cls  # <-- Note the / split /
            processed_class_dir.mkdir(parents=True, exist_ok=True)
            
            images = list(raw_class_dir.glob("*.jpg")) + list(raw_class_dir.glob("*.png"))
            print(f"Processing {len(images)} images for class: {split}/{cls}...")
            
            plot_saved = False
            for i, img_path in enumerate(images):
                img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
                if img is None: continue
                
                should_segment = cls not in ["Whitefly", "Leaf_Miner"]
                result = process_image(img, segment=should_segment)
                
                out_path = processed_class_dir / img_path.name
                cv2.imwrite(str(out_path), result["final"])
                
                if not plot_saved and split == "train":
                    plot_path = PLOTS_DIR / f"{cls}_evolution.png"
                    plot_feature_evolution(result, plot_path, cls)
                    plot_saved = True

if __name__ == "__main__":
    run_pipeline()
    print("✅ DIP Pipeline (Iteration 3) finished successfully!")
