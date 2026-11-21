"""Build a reproducible source-group split from original photos and annotated symptom regions."""

import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/corrected"
NAMES = ("Healthy Leaf", "Leaf Miner", "Red Mite", "Whitefly", "Yellow Mosaic")


def disk(relative):
    path = (ROOT / relative).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" else path


def photo(row):
    with Image.open(disk(row["path"])) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    if row["box"]:
        x, y, w, h = map(float, row["box"].split())
        width, height = image.size
        # Include surrounding leaf tissue around the annotated damage.
        image = image.crop(
            (
                max(0, int((x - w * 0.8) * width)),
                max(0, int((y - h * 0.8) * height)),
                min(width, int((x + w * 0.8) * width) + 1),
                min(height, int((y + h * 0.8) * height) + 1),
            )
        )
    return image


def build():
    base = disk("external_data")
    rows = []

    def add(path, label, source, group, box=""):
        rows.append(
            dict(
                path="external_data/" + path.relative_to(base).as_posix(),
                label=label,
                source=source,
                group=source + "/" + group,
                box=box,
            )
        )

    rng = random.Random(42)
    tea = sorted((base / "03-Red Spider-Tea-Disease").rglob("*_result.jpg"))
    mosaic = sorted(next(base.rglob("Cucumber Mosaic Virus")).glob("*.jpg"))
    for paths, label, source in [
        (tea, NAMES[2], "tea_original"),
        (mosaic, NAMES[4], "luffa_mosaic"),
    ]:
        for p in rng.sample(paths, min(250, len(paths))):
            add(p, label, source, p.stem)
    for p in sorted((base / "Healthy Leaves").rglob("*.jpg")):
        add(p, NAMES[0], "greenhouse", p.stem)
    rf = base / "roboflow_raw_data/dataset"
    for p in sorted(rf.glob("*/images/*.jpg")):
        annotation = p.parent.parent / "labels" / (p.stem + ".txt")
        if annotation.exists():
            ids = {
                line.split()[0]
                for line in annotation.read_text().splitlines()
                if line.strip()
            }
            if ids == {"4"}:
                add(p, NAMES[3], "roboflow", p.stem.split(".rf.")[0])
    field = base / "wmc4"
    for p in sorted((field / "healthy").glob("*.jpg")):
        add(p, NAMES[0], "wmc_healthy", p.stem)
    for p in sorted((field / "miner").glob("*.jpg")):
        lines = [
            s.split()
            for s in (field / "labels" / (p.stem + ".txt")).read_text().splitlines()
            if s.strip()
        ]
        boxes = [v[1:] for v in lines if len(v) == 5 and v[0] == "0"]
        boxes.sort(key=lambda v: float(v[2]) * float(v[3]), reverse=True)
        if boxes and all(v[0] == "0" for v in lines):
            add(p, NAMES[1], "wmc_miner", p.stem)
        for box in boxes[:2]:
            add(p, NAMES[1], "wmc_miner", p.stem, " ".join(box))
    # Keep related views together, including exact duplicates and visually close source frames.
    by_path = {r["path"]: r for r in rows}
    parent = {r["group"]: r["group"] for r in rows}

    def root(g):
        while g != parent[g]:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    seen = {}
    prior = defaultdict(list)
    for path, row in sorted(by_path.items()):
        g = row["group"]
        digest = hashlib.sha256(disk(path).read_bytes()).digest()
        if digest in seen:
            if seen[digest]["label"] != row["label"]:
                raise ValueError(f"Conflicting labels: {path}")
            parent[root(g)] = root(seen[digest]["group"])
        seen[digest] = row
        with Image.open(disk(path)) as im:
            thumb = np.array(im.convert("L").resize((32, 32)), np.float32)

        def phash(a):
            d = cv2.dct(np.ascontiguousarray(a))[:8, :8]
            return int.from_bytes(np.packbits(d > np.median(d)).tobytes(), "big")

        variants = [
            phash(np.rot90(s, i)) for s in (thumb, thumb[:, ::-1]) for i in range(4)
        ]
        for other, code in prior[row["label"]]:
            if min((v ^ code).bit_count() for v in variants) <= 6:
                parent[root(g)] = root(other)
        prior[row["label"]].append((g, variants[0]))
    strata = defaultdict(set)
    for r in rows:
        r["group"] = root(r["group"])
        strata[r["source"]].add(r["group"])
    split = {}
    for source, groups in sorted(strata.items()):
        groups = sorted(groups)
        if len(groups) < 3:
            raise ValueError(f"Insufficient independent groups for {source}")
        rng.shuffle(groups)
        n = max(1, round(len(groups) * 0.15))
        for i, g in enumerate(groups):
            part = "test" if i < n else "val" if i < 2 * n else "train"
            if g in split and split[g] != part:
                raise ValueError("Cross-source duplicate group")
            split[g] = part
    for r in rows:
        r["split"] = split[r["group"]]
    if any(
        {r["label"] for r in rows if r["split"] == s} != set(NAMES)
        for s in ("train", "val", "test")
    ):
        raise ValueError("Missing class; acquire the field data before building")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source_sha256.txt").write_text(
        "\n".join(sorted(d.hex() for d in seen)) + "\n"
    )
    audit = dict(
        views=len(rows),
        distinct_files=len(by_path),
        groups=len(split),
        pairwise_group_overlap=0,
        per_split={},
    )
    checked_groups = set()
    for part in ("train", "val", "test"):
        members = [r for r in rows if r["split"] == part]
        groups = {r["group"] for r in members}
        if groups & checked_groups:
            raise ValueError("Group overlap in split")
        checked_groups.update(groups)
        audit["per_split"][part] = dict(
            groups=len(groups),
            views_by_class=Counter(r["label"] for r in members),
            original_frames_by_source=Counter(
                r["source"] for r in members if not r["box"]
            ),
        )
    audit["known_limits"] = (
        "Plant and capture-session IDs were unavailable. Perceptual grouping only detects some related images."
    )
    (OUT / "split_audit.json").write_text(json.dumps(audit, indent=2))
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows
