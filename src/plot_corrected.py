"""Plot corrected validation/test results, class recall and matched storage totals."""

import csv
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.corrected_data import OUT, NAMES


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    names = list(metrics["test"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    x = np.arange(len(names))
    axes[0].bar(
        x - 0.18,
        [metrics["validation_macro_f1"][n] for n in names],
        0.36,
        label="Validation",
    )
    axes[0].bar(
        x + 0.18,
        [metrics["test"][n]["macro avg"]["f1-score"] for n in names],
        0.36,
        label="Test",
    )
    axes[0].set(
        xticks=x,
        xticklabels=names,
        ylabel="Macro-F1",
        ylim=(0, 1.05),
        title="Model comparison on grouped splits",
    )
    axes[0].legend()
    for i, name in enumerate(names):
        axes[1].bar(
            np.arange(5) + (i - 1) * 0.25,
            [metrics["test"][name][c]["recall"] for c in NAMES],
            0.25,
            label=name,
        )
    axes[1].set(
        xticks=np.arange(5),
        xticklabels=NAMES,
        ylabel="Recall",
        ylim=(0, 1.05),
        title="Test recall by class",
    )
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close(fig)
    with (OUT / "dip_stage_storage.csv").open() as f:
        stages = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    bars = ax.barh(
        [r["stage"] for r in stages], [int(r["bytes"]) / 1e6 for r in stages]
    )
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(
        xlabel="Total storage (decimal MB)",
        title=f"Matched full-frame test inputs (n={stages[0]['test_images']})",
    )
    ax.invert_yaxis()
    ax.margins(x=0.12)
    fig.savefig(OUT / "storage_by_stage.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
