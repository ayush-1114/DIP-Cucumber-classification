"""Extra experiments on corrected data. Run with the D-drive environment settings."""

import csv
import hashlib
import json
from importlib.metadata import version
from collections import Counter
from itertools import combinations

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.corrected_data import ROOT, OUT, NAMES, photo
from src.leaf_demo import PREPARE, prepare_rgb, load_model
from src.classical_ml.features import extract_features
from src.classical_ml.augment import augment_image
from src.classical_ml.feature_ablation import feature_columns

EXTRA = OUT / "extra_experiments"
CACHE = ROOT / ".runtime/extra_features.npz"


def save_csv(name, rows):
    with (EXTRA / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(actual, pred):
    recalls = recall_score(actual, pred, labels=range(5), average=None, zero_division=0)
    return dict(
        accuracy=accuracy_score(actual, pred),
        macro_f1=f1_score(
            actual, pred, labels=range(5), average="macro", zero_division=0
        ),
        **{f"{n} recall": r for n, r in zip(NAMES, recalls)},
    )


def weights(rows, indices):
    counts = Counter(rows[i]["group"] for i in indices)
    groups = {
        n: {rows[i]["group"] for i in indices if rows[i]["label"] == n} for n in NAMES
    }
    return np.array(
        [
            len(indices)
            / (5 * len(groups[rows[i]["label"]]) * counts[rows[i]["group"]])
            for i in indices
        ]
    )


def feature_key():
    paths = [
        "results/corrected/manifest.csv",
        "results/corrected/source_sha256.txt",
        "models/corrected_dip_mobilenet.pth",
        "src/dip/pipeline.py",
        "src/leaf_demo.py",
        "src/corrected_data.py",
        "src/classical_ml/features.py",
        "src/classical_ml/augment.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode())
        digest.update((ROOT / path).read_bytes())
    for package in (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "scikit-image",
        "pillow",
    ):
        digest.update((package + version(package)).encode())
    return digest.hexdigest()


def cache(rows, model):
    key = feature_key()
    if CACHE.exists():
        with np.load(CACHE) as saved:
            if str(saved["key"]) == key:
                return tuple(saved[name] for name in ("raw", "dip", "hand", "aug", "processed"))
    raw, dip, hand, augmented, processed = [], [], [], [], []
    rng = np.random.default_rng(42)
    for start in range(0, len(rows), 12):
        raw_batch, dip_batch = [], []
        for r in rows[start : start + 12]:
            rgb = photo(r)
            _, _, _, bgr = prepare_rgb(rgb)
            processed.append(bgr)
            raw_batch.append(PREPARE(rgb))
            dip_batch.append(
                PREPARE(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            )
            hand.append(extract_features(bgr))
            augmented.append(
                extract_features(augment_image(bgr, rng))
                if r["split"] == "train"
                else np.zeros_like(hand[-1])
            )
        with torch.inference_mode():
            raw.extend(model.features(torch.stack(raw_batch)).mean((2, 3)).numpy())
            dip.extend(model.features(torch.stack(dip_batch)).mean((2, 3)).numpy())
        print("Extra features", start + len(raw_batch), "/", len(rows), flush=True)
    CACHE.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        CACHE, key=key, raw=raw, dip=dip, hand=hand, aug=augmented, processed=processed
    )
    return map(np.asarray, (raw, dip, hand, augmented, processed))


def svm_fit(x, y, tr, va, rows):
    best = -1
    for c in (0.1, 1, 10):
        model = make_pipeline(
            StandardScaler(), SVC(C=c, decision_function_shape="ovr", break_ties=True)
        )
        model.fit(x[tr], y[tr], svc__sample_weight=weights(rows, tr))
        score = f1_score(y[va], model.predict(x[va]), average="macro")
        if score > best:
            best, chosen, result = score, c, model
    return result, chosen, best


def classical(rows, y, hand, aug, dip, tr, va, te):
    blocks = feature_columns()
    ablations = []
    models = {}
    for count in (1, 2, 3):
        for names in combinations(blocks, count):
            cols = np.concatenate([blocks[n] for n in names])
            model = RandomForestClassifier(
                n_estimators=300, class_weight=None, random_state=42, n_jobs=3
            )
            model.fit(hand[tr][:, cols], y[tr], sample_weight=weights(rows, tr))
            name = "+".join(names)
            result = dict(
                features=name,
                dimensions=len(cols),
                validation_macro_f1=f1_score(
                    y[va], model.predict(hand[va][:, cols]), average="macro"
                ),
                **metrics(y[te], model.predict(hand[te][:, cols])),
            )
            ablations.append(result)
            models[name] = model
    save_csv("rf_feature_ablation.csv", ablations)
    # Compare the same full-feature RF with and without one transformed copy per training view.
    w = weights(rows, tr)
    augmented_model = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=3
    )
    augmented_model.fit(
        np.concatenate((hand[tr], aug[tr])),
        np.tile(y[tr], 2),
        sample_weight=np.tile(w / 2, 2),
    )
    augmentation = []
    for name, model in [
        ("original", models["HOG+LBP+Colour"]),
        ("one augmentation per training view", augmented_model),
    ]:
        augmentation.append(
            dict(
                training=name,
                training_views=len(tr) * (2 if name != "original" else 1),
                validation_macro_f1=f1_score(
                    y[va], model.predict(hand[va]), average="macro"
                ),
                **metrics(y[te], model.predict(hand[te])),
            )
        )
    save_csv("augmentation_comparison.csv", augmentation)
    hybrid, c, val = svm_fit(dip, y, tr, va, rows)
    save_csv(
        "embedding_svm.csv",
        [
            dict(
                model="MobileNet embeddings + SVM",
                C=c,
                validation_macro_f1=val,
                **metrics(y[te], hybrid.predict(dip[te])),
            )
        ],
    )
    joblib.dump(hybrid, EXTRA / "embedding_svm.joblib", compress=3)
    curve = []
    leaf_groups = sorted({rows[i]["group"] for i in tr if y[i] == 1})
    for seed in (42, 43, 44):
        groups = np.random.default_rng(seed).permutation(leaf_groups)
        for fraction in (0.25, 0.5, 0.75, 1):
            chosen = set(groups[: max(1, round(fraction * len(groups)))])
            subset = np.array(
                [i for i in tr if y[i] != 1 or rows[i]["group"] in chosen]
            )
            model = make_pipeline(
                StandardScaler(),
                SVC(C=10, decision_function_shape="ovr", break_ties=True),
            )
            model.fit(hand[subset], y[subset], svc__sample_weight=weights(rows, subset))
            curve.append(
                dict(
                    seed=seed,
                    fraction=fraction,
                    leaf_groups=len(chosen),
                    leaf_views=int(sum(y[subset] == 1)),
                    validation_macro_f1=f1_score(
                        y[va], model.predict(hand[va]), average="macro"
                    ),
                    **metrics(y[te], model.predict(hand[te])),
                )
            )
    save_csv("leaf_miner_learning_curve.csv", curve)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, key, title in zip(
        axes,
        ("macro_f1", "Leaf Miner recall"),
        ("Test macro-F1", "Test Leaf Miner recall"),
    ):
        for seed in (42, 43, 44):
            points = [r for r in curve if r["seed"] == seed]
            ax.plot(
                [r["leaf_groups"] for r in points],
                [r[key] for r in points],
                marker="o",
                label=f"Seed {seed}",
            )
        ax.set(
            xlabel="Leaf Miner training source groups", ylabel=title, ylim=(0, 1)
        )
        ax.legend()
    fig.savefig(EXTRA / "learning_curve.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    coordinates = []
    for ax, x, name in zip(
        axes, (hand[te, :6084], dip[te]), ("HOG", "MobileNet features")
    ):
        reduced = PCA(n_components=min(30, len(te) - 1), random_state=42).fit_transform(
            x
        )
        points = TSNE(
            n_components=2,
            perplexity=20,
            random_state=42,
            init="pca",
            learning_rate="auto",
        ).fit_transform(reduced)
        for label in range(5):
            group = y[te] == label
            ax.scatter(*points[group].T, label=NAMES[label], s=20)
        ax.set_title(name)
        ax.legend(fontsize=8)
        coordinates.extend(
            dict(
                view=int(i),
                feature=name,
                label=NAMES[y[i]],
                x=float(p[0]),
                y=float(p[1]),
            )
            for i, p in zip(te, points)
        )
    fig.savefig(EXTRA / "feature_tsne.png", dpi=150)
    plt.close(fig)
    save_csv("tsne_coordinates.csv", coordinates)
    print("Classical experiments complete", flush=True)


def main():
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    EXTRA.mkdir(exist_ok=True)
    with (OUT / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    y = np.array([NAMES.index(r["label"]) for r in rows])
    parts = np.array([r["split"] for r in rows])
    tr, va, te = (np.flatnonzero(parts == part) for part in ("train", "val", "test"))
    model = load_model("corrected_dip_mobilenet.pth")
    raw, dip, hand, aug, processed = cache(rows, model)
    with torch.inference_mode():
        pred = model.classifier(torch.tensor(dip[te])).argmax(1).numpy()
    with (OUT / "test_predictions.csv").open() as f:
        expected = [NAMES.index(r["dip_mobilenet"]) for r in csv.DictReader(f)]
    assert (
        pred.tolist() == expected
    ), "Cached features must reproduce corrected test predictions"
    classical(rows, y, hand, aug, dip, tr, va, te)
    print("Saved experiment outputs", EXTRA, flush=True)


if __name__ == "__main__":
    main()
