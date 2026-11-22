"""Verify corrected experiment outputs and make their overview plot."""

import csv
import json
import numpy as np
import torch
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.corrected_data import OUT, ROOT, NAMES
from src.leaf_demo import load_model
from src.extra_corrected import EXTRA, CACHE


def read(name):
    with (EXTRA / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    torch.set_num_threads(2)
    with (OUT / "manifest.csv").open() as f:
        manifest = list(csv.DictReader(f))
    with (OUT / "test_predictions.csv").open() as f:
        reference = list(csv.DictReader(f))
    data = np.load(CACHE)
    test = np.array([i for i, r in enumerate(manifest) if r["split"] == "test"])
    for feature, name in (("raw", "raw_mobilenet"), ("dip", "dip_mobilenet")):
        model = load_model("corrected_" + name + ".pth")
        with torch.inference_mode():
            pred = model.classifier(torch.tensor(data[feature][test])).argmax(1).numpy()
        assert [NAMES[i] for i in pred] == [r[name] for r in reference]
    svm = joblib.load(ROOT / "models/corrected_dip_svm.joblib")
    assert [NAMES[i] for i in svm.predict(data["hand"][test])] == [
        r["dip_svm"] for r in reference
    ]
    stress = read("stress_predictions.csv")
    assert len(stress) == 135 * 5 * 2
    for model in ("raw", "dip"):
        clean = [
            r["prediction"]
            for r in stress
            if r["model"] == model and r["condition"] == "clean"
        ]
        assert clean == [r[model + "_mobilenet"] for r in reference]
    ablation = read("rf_feature_ablation.csv")
    selected = max(ablation, key=lambda r: float(r["validation_macro_f1"]))
    curve = read("leaf_miner_learning_curve.csv")
    localization = read("leaf_miner_localization.csv")
    checks = read("gradcam_checks.csv")
    unique = {(r["view"], r["model"], r["target"]): r for r in checks}
    result = dict(
        verified_test_views=135,
        feature_and_clean_predictions_reproduced=True,
        rf_validation_selected=selected,
        cam_deletion={},
    )
    for model in ("raw", "dip"):
        members = [r for r in unique.values() if r["model"] == model]
        result["cam_deletion"][model] = dict(
            distinct_view_targets=len(members),
            hot_deletion_exceeds_random=sum(
                float(r["top20_deletion_score_drop"]) > float(r["random20_mean_drop"])
                for r in members
            ),
        )
    for source in ("stress_summary.csv", "legacy_tensor_noise_summary.csv"):
        rows = read(source)
        result[source] = [
            dict(
                model=m,
                condition=c,
                mean_accuracy=float(
                    np.mean(
                        [
                            float(r["accuracy"])
                            for r in rows
                            if r["model"] == m
                            and r.get("condition", "tensor_noise_std0.8") == c
                        ]
                    )
                ),
            )
            for m in ("raw", "dip")
            for c in sorted({r.get("condition", "tensor_noise_std0.8") for r in rows})
        ]
    result["localization"] = [
        dict(
            model=m,
            view_type=t,
            views=len(g),
            mean_box_enrichment=float(
                np.mean([float(r["enrichment_vs_uniform"]) for r in g])
            ),
            peak_in_box=sum(r["peak_in_damage_box"] == "True" for r in g),
        )
        for m in ("raw", "dip")
        for t in ("crop", "full_frame")
        if (g := [r for r in localization if r["model"] == m and r["view_type"] == t])
    ]
    result["learning_curve_mean_leaf_recall"] = {
        str(f): float(
            np.mean(
                [
                    float(r["Leaf Miner recall"])
                    for r in curve
                    if float(r["fraction"]) == f
                ]
            )
        )
        for f in (0.25, 0.5, 0.75, 1)
    }
    (EXTRA / "verification_summary.json").write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    x = np.arange(len(ablation))
    axes[0].bar(
        x - 0.18,
        [float(r["validation_macro_f1"]) for r in ablation],
        0.36,
        label="Validation",
    )
    axes[0].bar(x + 0.18, [float(r["macro_f1"]) for r in ablation], 0.36, label="Test")
    axes[0].set(
        xticks=x,
        xticklabels=[r["features"] for r in ablation],
        ylabel="Macro-F1",
        title="RF feature ablation",
        ylim=(0, 1),
    )
    axes[0].tick_params(axis="x", rotation=65)
    axes[0].legend()
    for offset, file, label in (
        (-0.18, "augmentation_comparison.csv", "RF"),
        (0.18, "svm_augmentation.csv", "SVM"),
    ):
        rows = read(file)
        axes[1].bar(
            np.arange(2) + offset,
            [float(r["macro_f1"]) for r in rows],
            0.36,
            label=label,
        )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Original", "With augmentation"],
        ylabel="Test macro-F1",
        ylim=(0, 1),
        title="Training-only augmentation",
    )
    axes[1].legend()
    rows = read("quantized_storage.csv")
    axes[2].bar([0, 1], [float(r["bytes"]) / 1e6 for r in rows])
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["FP32", "INT8 weight storage"],
        ylabel="Checkpoint size (decimal MB)",
        title="Weight-storage comparison",
    )
    fig.savefig(EXTRA / "extra_comparisons.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
