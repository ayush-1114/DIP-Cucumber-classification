# Project updates

This file records changes made by Codex in DIP-project. Subsequent work should append an entry describing changed files, purpose, verification, results, pending work and any GitHub commit.

## 22 November 2026 — Retrospective record of initial setup and demo work

This entry records work completed in the preceding chat turns. The log was added after the user pointed out its omission.

### Repository setup — 5c0d35d

- Copied the current MLEE working folder into DIP-project inside the DIP Project workspace, including local datasets and the virtual environment.
- Preserved existing Git history and created the private ayush-1114/DIP-Cucumber-classification repository.
- Added reports/DIP_PROJECT_STATUS.md with the assignment requirements, completed work, extra experiments and remaining work.
- Committed the existing assignment images, MLEE report, saved evaluation outputs and two MobileNetV2 checkpoints. These were present locally before the copy.
- Bulk datasets and the virtual environment are retained locally and excluded from GitHub by the existing ignore rules.
- Verified all 72,411 source and copied files matched by relative path and size. Verified the pushed main commit.

### DIP demo and evidence — 76826eb

New executable files

- src/leaf_demo.py loads a single photo, handles orientation, applies the existing DIP pipeline without consulting a class label, predicts with the saved processed-image MobileNetV2 and saves a labelled stage panel and JPEG.
- src/evaluate_leaf_demo.py evaluates the live path against the manifest test images and writes predictions, class metrics, a confusion matrix and segmentation decision counts.
- src/dip_stage_storage.py compares original storage with resize/encode, CLAHE and foreground masking at JPEG quality 95.
- run_dip.ps1 runs the single-image demo and both evaluation scripts after environment and data setup.

Documentation changes

- Updated README.md with links to the DIP documents.
- Updated reports/DIP_PROJECT_STATUS.md to record the new work and remaining requirements.
- Added reports/PRESENTATION_GUIDE.md with photo capture, commands, viva explanations and rehearsal instructions.
- Added reports/DIP_TECHNICAL_REPORT.md with DIP theory, flow, stage visualization, baseline comparisons, storage measurements and limitations.

Generated artifacts

- results/demo/healthy_0009/processed.jpg and stages.png
- results/metrics/demo_predictions.csv and demo_classification_report.txt
- results/metrics/dip_stage_storage.csv

Verification and results

- Ran the complete run_dip.ps1 workflow successfully with the healthy_0009.jpg sample.
- Inspected the rendered processing panel.
- Live workflow correctly classified 132 of 151 held-out images, yielding 87.42% accuracy and macro-F1 0.8427.
- Of 19 errors, 14 were Whitefly classified as Leaf Miner and three were Whitefly classified as Red Mite.
- Across the same 151 images, original files totalled 13,730,467 bytes, resize-only JPEGs 2,362,055 bytes, CLAHE JPEGs 3,103,438 bytes and masked CLAHE JPEGs 2,677,892 bytes.
- Pushed commit 76826eb to the private repository.

### Existing code and current limitations

The demo reuses src/dip/pipeline.py. The existing pipeline, training scripts, original model weights and existing dataset were retained. The newly added live path uses image-based segmentation for every class, while the historical batch pipeline has class-specific exceptions.

The live predictor uses the processed-image checkpoint. Whole-leaf performance on independently photographed cucumber leaves remains unverified. Whitefly errors, proxy-host data, tight pest crops and small minority test sets require discussion and rehearsal. Final report formatting, student identifiers, citation finalization and fresh-install reproducibility remain pending.

### Logging correction

Created this updates.md file in the repository root. Future updates should be recorded here alongside implementation changes.
