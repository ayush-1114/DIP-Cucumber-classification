"""Controlled SVM augmentation comparison using the corrected feature cache."""

import csv
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from src.corrected_data import OUT, NAMES
from src.extra_corrected import CACHE, save_csv, metrics, weights


def main():
    with (OUT / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    data = np.load(CACHE)
    x, aug = data["hand"], data["aug"]
    y = np.array([NAMES.index(r["label"]) for r in rows])
    parts = np.array([r["split"] for r in rows])
    tr, va, te = (np.flatnonzero(parts == p) for p in ("train", "val", "test"))
    output = []
    for augmented in (False, True):
        xx = np.concatenate((x[tr], aug[tr])) if augmented else x[tr]
        yy = np.tile(y[tr], 2) if augmented else y[tr]
        w = np.tile(weights(rows, tr) / 2, 2) if augmented else weights(rows, tr)
        best = -1
        for c in (0.1, 1, 10):
            model = make_pipeline(
                StandardScaler(),
                SVC(C=c, decision_function_shape="ovr", break_ties=True),
            )
            model.fit(xx, yy, svc__sample_weight=w)
            val = f1_score(y[va], model.predict(x[va]), average="macro")
            if val > best:
                best, chosen, selected = val, c, model
        output.append(
            dict(
                augmented=augmented,
                C=chosen,
                validation_macro_f1=best,
                **metrics(y[te], selected.predict(x[te]))
            )
        )
    save_csv("svm_augmentation.csv", output)
    print(output)


if __name__ == "__main__":
    main()
