# Repository guide

This D-drive copy uses .fresh-env. Run .\verify_setup.ps1 for the checked environment and D-drive cache settings. Start with the demo command below. This guide describes the current files as of 22 November 2026.

## Everyday commands

Put unlabelled photos directly in demo_input, then run the demo.

```powershell
python src/leaf_demo.py
```

Open results/demo/predictions.csv in Excel. It contains one row per input photo and all five class scores. Close Excel before rerunning so Windows permits the CSV to be refreshed. Individual folders in results/demo contain processed.jpg and stages.png.

To check saved model performance on the corrected benchmark, run these commands.

```powershell
python -m src.evaluate_leaf_demo
python -m src.evaluate_leaf_demo --model dip_svm
```

To rebuild the corrected data manifest, retrain all three models and regenerate benchmark outputs, run the following command after the source datasets are installed.

```powershell
.\rebuild_dip.ps1
```

Rebuilding replaces corrected checkpoints and measurements. Save a versioned copy before intentionally starting a new experiment.

## Current workflow

1. external_data holds source photographs and annotations.
2. corrected_data.py groups related photos, makes the split and loads full frames or annotated symptom regions.
3. train_corrected.py uses the shared DIP preparation and trains raw MobileNet, DIP MobileNet and DIP SVM.
4. models holds saved trained models. results/corrected holds their manifest, measurements and plots.
5. leaf_demo.py loads the selected corrected model and predicts photos in demo_input.

The current MobileNet models have frozen ImageNet feature extractors and newly trained five-class heads. Validation chooses the demo model. The current choice is DIP MobileNet.

## Files used by the current workflow

- src/leaf_demo.py runs prediction, exports the single CSV, and renders the original, padding, masks, enhancement, Otsu and edge panels. It also provides shared preprocessing and model-loading functions used by training and evaluation.
- src/dip/pipeline.py implements resize/padding, HSV segmentation seeds, morphology, GrabCut, CLAHE, Otsu and Canny. Its process_image function is used by the corrected workflow. Its standalone batch runner still belongs to the earlier data/raw workflow.
- src/corrected_data.py defines the current class order, source grouping, split, manifest and photo loading. Related symptom crops stay with their source photo.
- src/fetch_field_data.py downloads the selected WMC-Leafset v4 field images and annotations and verifies checksums. Other source datasets must already be present.
- src/train_corrected.py extracts features, fits classifier heads and SVM, selects settings using validation, saves models, and writes test metrics, predictions and confusion matrices.
- src/classical_ml/features.py extracts HOG, LBP and HSV features. The corrected SVM actively depends on it.
- src/evaluate_leaf_demo.py runs the deployed model on corrected test inputs and checks that predictions match the training evaluation. It also reports the full-frame subset.
- src/dip_stage_storage.py compares byte totals on the same full-frame test images after resizing, enhancement and masking at fixed JPEG quality.
- src/plot_corrected.py draws model comparisons and storage plots from corrected results.
- src/evaluate_field_set.py evaluates an independent labelled CSV with path,label,source columns. It saves predictions and five scores, and rejects exact benchmark duplicates.
- rebuild_dip.ps1 runs download, corrected training, deployment verification, storage measurement and plotting.
- run_dip.ps1 runs a demo, then benchmark verification and storage measurement. Use leaf_demo.py alone for a quick demonstration.
- requirements.txt lists Python dependencies. results/corrected/runtime_versions.json records versions used in the measured run.

## Folders and root documents

- demo_input contains the photos you want to predict. Class folders are unnecessary.
- external_data contains large original datasets, omitted from Git. The local wmc4 folder is a junction to the D-drive cache recorded in updates.md. A fresh checkout needs source-data setup.
- data contains the earlier raw/processed train, validation and test layout, historical manifests and any feature caches. The corrected workflow reads sources through its own manifest.
- models contains both corrected and earlier checkpoints.
- results/corrected contains current benchmark evidence and separately named diagnostic outputs.
- results/demo contains generated demo CSV and per-photo panels.
- results/metrics, results/figures and results/storage_reduction contain earlier measurements and supporting experiments. Their checkpoints and splits differ from the corrected run.
- reports contains existing reports, source notes, progress records and the presentation guide.
- notebooks is available for exploratory notebooks. The current commands run through Python modules.
- README.md provides setup and workflow instructions.
- updates.md records changes and verification.
- guidelines.png and project-task.jpeg are the course requirements.

## Model files

- corrected_dip_mobilenet.pth is the current default demo checkpoint.
- corrected_raw_mobilenet.pth is the raw-input comparison checkpoint.
- corrected_dip_svm.joblib is the corrected handcrafted-feature SVM.
- mobilenetv2_cucumber.pth and mobilenetv2_raw.pth belong to earlier training and associated experiments.

The corrected class order is Healthy Leaf, Leaf Miner, Red Mite, Whitefly, Yellow Mosaic. Legacy configuration and folder names differ. The current class order comes from corrected_data.py.

## Current result files

- manifest.csv lists source paths, labels, groups, crop boxes and splits.
- split_audit.json records split counts and detected group overlap.
- source_sha256.txt holds source-image hashes used to reject exact benchmark copies in independent evaluation.
- metrics.json holds validation/model selection, per-class test metrics and aggregate storage.
- test_predictions.csv records each held-out view and all three model predictions.
- deployed_test_report.txt verifies the chosen demo model and includes full-frame metrics.
- deployed_dip_svm_report.txt verifies the deployed SVM.
- confusion_matrices.png and model_comparison.png compare corrected models.
- dip_stage_storage.csv and storage_by_stage.png show matched storage measurements.
- runtime_versions.json records the training environment.
- external_diagnostic.csv records the nine Google-photo predictions against provisional labels.
- segmentation_diagnostic.csv compares four fixed masking variants on those nine photos.

## Earlier experiment code

These files support the initial experiments and their outputs. They use the earlier datasets/checkpoints and need adaptation before generating evidence for corrected models.

- train_dl.py and train_dl_raw.py train the earlier processed and raw MobileNet models.
- evaluate_dl.py and evaluate_dl_raw.py evaluate those models.
- gradcam.py and gradcam_raw.py render attribution maps for earlier models.
- background_swap.py runs the earlier clean/noisy comparison. Its perturbation adds Gaussian noise across the whole normalized input after preprocessing.
- storage_metric.py measures earlier folder storage.
- organize_datasets.py assembles the earlier raw training folders.
- crop_pests.py extracts YOLO insect boxes for the earlier dataset.
- downsample_data.py reduces earlier training folders.
- data_prep/split_dataset.py builds the earlier shared split.
- config.py contains the earlier folder configuration and class-name mappings.
- classical_ml/train.py trains earlier SVM/Random Forest baselines.
- classical_ml/augment.py performs training augmentation for the earlier classical workflow.
- classical_ml/feature_ablation.py compares feature combinations.
- classical_ml/leaf_miner_learning_curve.py studies sample count.
- classical_ml/mobilenet_svm.py fits SVM to learned embeddings.
- classical_ml/plot_feature_tsne.py visualizes feature embeddings.
- classical_ml/measure_quantized_storage.py measures compressed checkpoint storage.
- __init__.py files mark Python packages.

## Diagnostic and cleanup candidates

- check_demo_photos.py rechecks the nine locally supplied Google photos using provisional filename-derived labels.
- diagnose_segmentation.py compares deployed masks, full frames, HSV masks and a fixed central GrabCut attempt. Its results support debugging.
- results/demo_input_check and results/demo_score_check contain generated rehearsal/debugging runs.
- results/google_photo_contact_sheet.jpg and results/roboflow_source_audit.jpg are visual inspection aids.
- Python __pycache__ folders are regenerated automatically.

Generated panels and debugging runs can be regenerated from the images and scripts. Source photos, model files, manifests and experiment evidence have distinct roles and require review before removal. Grad-CAM and robustness artifacts substantiate the extra work already completed.

## Requirement completion

The requested DIP operations, classifiers, storage measurements, class metrics, plots and demo export exist. The earlier work also includes Grad-CAM, noise robustness, ablation, learning curves and checkpoint-storage experiments.

The corrected-model robustness and Grad-CAM experiments still need rerunning. A fresh-install end-to-end test is pending. GrabCut can retain backgrounds, proxy hosts limit applicability, and the corrected demo matched only 3/9 provisional Google-photo labels. Most implementation components exist, while the intended reliable field-leaf outcome remains partially satisfied.

Use the corrected results for the current models and identify earlier experiments with their original checkpoints and split. The final submission report and packaging are pending. This guide makes no file deletions or directory moves.


## Corrected extra experiments, completed 22 November 2026

The extra studies have now been rerun for the corrected split and models. See results/corrected/extra_experiments/README.md for methods, output descriptions and limitations. Use rerun_extras.ps1 to reproduce them. New modules are extra_corrected.py, extra_svm_augmentation.py, extra_neural.py, check_cam_localization.py, extra_legacy_noise.py and summarize_extras.py. Earlier files listed above remain historical implementations. The updated results include mixed and negative findings, especially for stress robustness and full-frame CAM localization.
