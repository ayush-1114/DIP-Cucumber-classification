"""Train the classical ML baselines: SVM and Random Forest (roadmap Phase 3, Track 1).

Run from the project root, inside mlee-env:

    python -m src.classical_ml.train                  # Dataset B (data/processed) - the real run
    python -m src.classical_ml.train --dataset raw    # raw images - only a code check
    python -m src.classical_ml.train --no-augment     # without augmentation, to compare

IMPORTANT: the results that go in the report must come from the DIP-processed images
(Dataset B, produced by AP's pipeline). Numbers from --dataset raw are NOT comparable
with the deep-learning track.

Steps
-----
1. Extract HOG + LBP + colour features for train / val / test
   (cached in data/features/ so a second run is fast).
   The small classes in TRAIN are topped up with augmented copies
   (see src/classical_ml/augment.py). Val and test are never augmented.
2. For each model, try a few settings: fit on TRAIN, score on VAL (macro-F1),
   keep the best one. The TEST set is never used for choosing.
3. Score the chosen model once on TEST.
4. Save for PJ's evaluation:
     models/svm_model.pkl and models/rf_model.pkl   (the real run on Dataset B;
                                                     other runs: <model>_model_<tag>.pkl)
     results/metrics/<model>_<tag>_test_predictions.csv   (filename, y_true, y_pred)
     results/metrics/classical_ml_<tag>_summary.json      (scores, settings, timing)
   <tag> is the dataset name, plus "_noaug" when run with --no-augment.

Class imbalance is handled three ways: augmentation of the small classes,
class_weight="balanced", and comparing models with macro-F1 (every class counts
equally) rather than plain accuracy.
"""

import argparse
import csv
import json
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.classical_ml.augment import augmented_training_features
from src.classical_ml.features import extract_features, extract_folder, load_image

FEATURES_DIR = config.ROOT_DIR / "data" / "features"
METRICS_DIR = config.RESULTS_DIR / "metrics"
DATASET_DIRS = {"processed": config.PROCESSED_DIR, "raw": config.RAW_DIR}


def build_svm(C):
    # Scale -> PCA (keep 95% of the variance; HOG alone is 6084 values) -> RBF SVM
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95, random_state=config.SEED),
        SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced"),
    )


def build_rf(max_depth):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=config.SEED,
        n_jobs=-1,
    )


# model name -> (builder function, list of settings to try on the validation set)
MODELS = {
    "svm": (build_svm, [{"C": 0.1}, {"C": 1}, {"C": 10}, {"C": 100}]),
    "rf": (build_rf, [{"max_depth": None}, {"max_depth": 10}, {"max_depth": 20}]),
}


def load_split(dataset, split, use_cache):
    """Return X, y, filenames for one split, extracting features if not cached."""
    cache = FEATURES_DIR / f"{dataset}_{split}.npz"
    if use_cache and cache.exists():
        data = np.load(cache, allow_pickle=False)
        return data["X"], data["y"], list(data["filenames"])

    X, y, paths = extract_folder(DATASET_DIRS[dataset] / split, desc=f"{dataset}/{split}")
    if len(y) == 0:
        hint = "Run the DIP pipeline if needed, then: python -m src.data_prep.split_dataset"
        raise SystemExit(f"No images in {DATASET_DIRS[dataset] / split}. {hint}")
    filenames = [f"{p.parent.name}/{p.name}" for p in paths]
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, filenames=np.array(filenames))
    return X, y, filenames


def load_augmented(dataset, use_cache):
    """Return X_extra, y_extra, {class: copies added} for the augmented train images."""
    cache = FEATURES_DIR / f"{dataset}_train_augmented.npz"
    if use_cache and cache.exists():
        data = np.load(cache, allow_pickle=False)
        y_extra = data["y"]
        added = {cls: int((y_extra == idx).sum()) for cls, idx in config.CLASS_TO_IDX.items()
                 if (y_extra == idx).any()}
        return data["X"], y_extra, added

    X_extra, y_extra, added = augmented_training_features(DATASET_DIRS[dataset] / "train")
    if len(y_extra):
        np.savez_compressed(cache, X=X_extra, y=y_extra)
    return X_extra, y_extra, added


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=range(len(config.CLASSES)),
                    average="macro", zero_division=0)


def select_and_train(name, X_train, y_train, X_val, y_val):
    """Fit every setting on train, return (best model, best setting, best val macro-F1)."""
    builder, settings = MODELS[name]
    best = (None, None, -1.0)
    for params in settings:
        model = builder(**params).fit(X_train, y_train)
        score = macro_f1(y_val, model.predict(X_val))
        print(f"  {name} {params}: val macro-F1 = {score:.4f}")
        if score > best[2]:
            best = (model, params, score)
    return best


def time_inference(model, dataset, test_filenames, n_images=30):
    """Average ms per image for the full path: read image -> features -> predict."""
    rng = np.random.default_rng(config.SEED)
    chosen = rng.choice(len(test_filenames), size=min(n_images, len(test_filenames)), replace=False)
    paths = [DATASET_DIRS[dataset] / "test" / test_filenames[i] for i in chosen]
    start = time.perf_counter()
    for path in paths:
        model.predict(extract_features(load_image(path)).reshape(1, -1))
    return (time.perf_counter() - start) / len(paths) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASET_DIRS), default="processed")
    parser.add_argument("--no-cache", action="store_true",
                        help="re-extract features even if a cached copy exists")
    parser.add_argument("--no-augment", action="store_true",
                        help="do not add augmented copies of the small classes")
    args = parser.parse_args()

    if args.dataset == "raw":
        print("NOTE: training on RAW images. Use this only to check the code; "
              "report results must come from --dataset processed.\n")

    X_train, y_train, _ = load_split(args.dataset, "train", not args.no_cache)
    X_val, y_val, _ = load_split(args.dataset, "val", not args.no_cache)
    X_test, y_test, test_files = load_split(args.dataset, "test", not args.no_cache)

    tag = f"{args.dataset}_noaug" if args.no_augment else args.dataset
    added = {}
    if not args.no_augment:
        X_extra, y_extra, added = load_augmented(args.dataset, not args.no_cache)
        if len(y_extra):
            X_train = np.concatenate([X_train, X_extra])
            y_train = np.concatenate([y_train, y_extra])
        print(f"Augmented copies added to train: {added or 'none'}")
    print(f"Features: {X_train.shape[1]} per image | "
          f"train {len(y_train)}, val {len(y_val)}, test {len(y_test)}\n")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": args.dataset, "classes": config.CLASSES,
               "n_features": int(X_train.shape[1]),
               "augmented_copies_added": added, "models": {}}

    for name in MODELS:
        print(f"Training {name.upper()}...")
        model, params, val_f1 = select_and_train(name, X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = macro_f1(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, labels=range(len(config.CLASSES)),
            target_names=config.CLASSES, zero_division=0, output_dict=True)
        ms_per_image = time_inference(model, args.dataset, test_files)
        print(f"  best {params} | test accuracy = {test_acc:.4f}, "
              f"test macro-F1 = {test_f1:.4f}, {ms_per_image:.1f} ms/image\n")

        model_file = f"{name}_model.pkl" if tag == "processed" else f"{name}_model_{tag}.pkl"
        joblib.dump(model, config.MODELS_DIR / model_file)
        with open(METRICS_DIR / f"{name}_{tag}_test_predictions.csv",
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "y_true", "y_pred"])
            for filename, true, pred in zip(test_files, y_test, y_pred):
                writer.writerow([filename, config.IDX_TO_CLASS[int(true)],
                                 config.IDX_TO_CLASS[int(pred)]])

        summary["models"][name] = {
            "best_params": params,
            "val_macro_f1": round(float(val_f1), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_f1": round(float(test_f1), 4),
            "inference_ms_per_image": round(float(ms_per_image), 2),
            "per_class": {cls: {k: round(float(v), 4) for k, v in report[cls].items()}
                          for cls in config.CLASSES},
        }

    summary_path = METRICS_DIR / f"classical_ml_{tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved models to {config.MODELS_DIR}")
    print(f"Saved predictions and summary to {METRICS_DIR}")


if __name__ == "__main__":
    main()
