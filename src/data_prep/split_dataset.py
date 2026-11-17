"""One shared train/val/test split for the whole team (roadmap step 1.5).

The split is defined by ONE committed file, data/split_manifest.csv. It lists every
image we use and whether it is train, val or test. Everyone applies that same list,
so the classical ML (SN), deep learning (AP) and evaluation (PJ) all use identical
images. Nobody should call train_test_split on their own.

Run from the project root, inside mlee-env:

    python -m src.data_prep.split_dataset              # apply the list to data/raw AND data/processed
    python -m src.data_prep.split_dataset --dry-run    # only print what would happen
    python -m src.data_prep.split_dataset --rebuild    # (SN only) recreate the list from data/raw

Applying the list (what teammates run)
--------------------------------------
For data/raw and data/processed, every image found in train/, val/, test/, _unused/
or train_backup/ is matched to the list by class + file name and moved to
<dataset>/<split>/<Class>/. Images not in the list (duplicates, images dropped by
downsampling) go to <dataset>/_unused/<Class>/. Nothing is deleted, and running it
again changes nothing.

How the list is built (--rebuild)
---------------------------------
1. Pool every raw image, including any earlier split, _unused/ and train_backup/.
2. Drop exact duplicate files (same bytes, MD5 hash).
3. Downsample every class to at most MAX_PER_CLASS images (team decision, replaces
   src/downsample_data.py, whose result depended on the machine it ran on).
4. Split each class 70/15/15 (stratified), keeping all crops of one Roboflow photo in
   the same split so near-identical crops cannot leak from train into test.
Steps 3 and 4 use the shared SEED from src/config.py and sorted file names, so the
list is the same on every machine.
"""

import argparse
import csv
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict

from src import config

IMG_EXTS = {".jpg", ".jpeg", ".png"}
MAX_PER_CLASS = 300          # same cap as src/downsample_data.py; None = keep everything
UNUSED = "_unused"
# Folders an image may currently be in (the last two are from earlier scripts)
POOL_DIRS = [*config.SPLITS, UNUSED, "train_backup", "_duplicates"]
DATASETS = {"raw": config.RAW_DIR, "processed": config.PROCESSED_DIR}
MANIFEST_PATH = config.ROOT_DIR / "data" / "split_manifest.csv"
SUMMARY_PATH = config.RESULTS_DIR / "metrics" / "dataset_split_summary.csv"

# Roboflow crops are named "<rf-split>_<photo name>_crop<N>.jpg" by src/crop_pests.py
CROP_PATTERN = re.compile(r"^(?:train|valid|test)_(?P<photo>.+)_crop\d+$")


def sort_key(item):
    cls, path = item
    return cls, path.name.lower(), path.name


def collect_images(base_dir):
    """Every image under base_dir/<pool dir>/<Class>/ as a sorted list of (class, path)."""
    images = []
    for folder in POOL_DIRS:
        for cls in config.CLASSES:
            cls_dir = base_dir / folder / cls
            if cls_dir.is_dir():
                images += [(cls, p) for p in cls_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(images, key=sort_key)


def match_key(cls, path):
    """Images are matched to the list by class + file name without extension."""
    return cls, path.stem.lower()


def group_id(cls, path):
    """Images sharing a group id must end up in the same split."""
    match = CROP_PATTERN.match(path.stem)
    return f"{cls}/{match.group('photo') if match else path.stem}"


# ---------------------------------------------------------------- building the list

def drop_duplicates(images):
    """Keep the first file of every set of byte-identical files."""
    seen, unique = {}, []
    for cls, path in images:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            if seen[digest] != cls:
                print(f"WARNING: {path.name} ({cls}) is identical to an image in {seen[digest]}")
            continue
        seen[digest] = cls
        unique.append((cls, path))
    return unique


def downsample(images):
    """At most MAX_PER_CLASS images per class, chosen with the shared seed."""
    if MAX_PER_CLASS is None:
        return images
    kept = []
    for cls in config.CLASSES:
        cls_images = [item for item in images if item[0] == cls]
        if len(cls_images) > MAX_PER_CLASS:
            rng = random.Random(f"{config.SEED}-downsample-{cls}")
            cls_images = sorted(rng.sample(cls_images, MAX_PER_CLASS), key=sort_key)
        kept += cls_images
    return kept


def assign_splits(images):
    """Return {path: split}: stratified 70/15/15 that keeps groups together.

    Done class by class (that is what makes it stratified). Inside a class the groups
    are shuffled with the shared SEED, then handed out one at a time, biggest first,
    to whichever split is currently furthest below its target share.
    """
    groups = defaultdict(list)
    for cls, path in images:
        groups[group_id(cls, path)].append(path)

    assignment = {}
    for cls in config.CLASSES:
        cls_groups = sorted(gid for gid in groups if gid.split("/", 1)[0] == cls)
        random.Random(f"{config.SEED}-{cls}").shuffle(cls_groups)
        cls_groups.sort(key=lambda gid: len(groups[gid]), reverse=True)  # stable sort

        total = sum(len(groups[gid]) for gid in cls_groups)
        targets = {s: ratio * total for s, ratio in zip(config.SPLITS, config.SPLIT_RATIOS)}
        filled = dict.fromkeys(config.SPLITS, 0)
        for gid in cls_groups:
            split = max(config.SPLITS, key=lambda s: (targets[s] - filled[s]) / targets[s])
            filled[split] += len(groups[gid])
            for path in groups[gid]:
                assignment[path] = split
    return assignment


def rebuild_manifest():
    images = collect_images(config.RAW_DIR)
    if not images:
        raise SystemExit(f"No images found under {config.RAW_DIR}. Extract raw.zip first.")
    unique = drop_duplicates(images)
    kept = downsample(unique)
    assignment = assign_splits(kept)

    keys = [match_key(cls, path) for cls, path in kept]
    if len(set(keys)) != len(keys):
        raise SystemExit("Two kept images share a class and file name; cannot build the list.")

    rows = sorted((assignment[path], cls, path.name, group_id(cls, path)) for cls, path in kept)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "filename", "group"])
        writer.writerows(rows)

    found, n_unique = Counter(c for c, _ in images), Counter(c for c, _ in unique)
    counts = Counter((cls, assignment[path]) for cls, path in kept)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", *config.SPLITS, "total",
                         "duplicates_dropped", "downsampled_out"])
        for cls in config.CLASSES:
            row = [counts[(cls, s)] for s in config.SPLITS]
            writer.writerow([cls, *row, sum(row), found[cls] - n_unique[cls],
                             n_unique[cls] - sum(row)])
    print(f"Rebuilt the list from {len(images)} raw images: {len(unique)} unique, "
          f"{len(kept)} kept.\n  {MANIFEST_PATH}\n  {SUMMARY_PATH}\n")


# ---------------------------------------------------------------- applying the list

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found. Run 'git pull origin main' first.")
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return {(row["class"], row["filename"].rsplit(".", 1)[0].lower()): row["split"]
                for row in csv.DictReader(f)}


def apply_manifest(name, manifest, dry_run):
    base_dir = DATASETS[name]
    images = collect_images(base_dir)
    if not images:
        print(f"[{name}] no images in {base_dir} - skipped.\n")
        return

    moves, counts, found = [], Counter(), set()
    for cls, path in images:
        key = match_key(cls, path)
        split = manifest.get(key) if key not in found else None   # 2nd copy of a name -> unused
        if split:
            found.add(key)
        counts[(cls, split or UNUSED)] += 1
        dest = base_dir / (split or UNUSED) / cls / path.name
        if dest != path:
            moves.append((path, dest))
    missing = sorted(set(manifest) - found)

    columns = [*config.SPLITS, UNUSED]
    print(f"[{name}] {base_dir}")
    print(f"  {'Class':<15}" + "".join(f"{c:>9}" for c in columns))
    for cls in config.CLASSES:
        print(f"  {cls:<15}" + "".join(f"{counts[(cls, c)]:>9}" for c in columns))
    print(f"  {'TOTAL':<15}" + "".join(
        f"{sum(counts[(cls, c)] for cls in config.CLASSES):>9}" for c in columns))

    if missing:
        print(f"  WARNING: {len(missing)} listed images are not in this folder yet, e.g. "
              + ", ".join(f"{c}/{n}" for c, n in missing[:3]))
        if name == "processed":
            print("  -> run the DIP pipeline on the full raw set, then run this script again.")
        else:
            print("  -> extract the full raw.zip (or restore train_backup), then run this again.")

    if dry_run:
        print(f"  Dry run: {len(moves)} files would be moved.\n")
        return
    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    print(f"  Moved {len(moves)} files.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without moving any file")
    parser.add_argument("--rebuild", action="store_true",
                        help="recreate data/split_manifest.csv from data/raw (SN only; "
                             "changes the split for the whole team)")
    args = parser.parse_args()

    if args.rebuild:
        rebuild_manifest()
    manifest = load_manifest()
    print(f"Shared list: {len(manifest)} images "
          f"({dict(sorted(Counter(manifest.values()).items()))})\n")
    for name in (DATASETS if args.dataset == "both" else [args.dataset]):
        apply_manifest(name, manifest, args.dry_run)


if __name__ == "__main__":
    main()
