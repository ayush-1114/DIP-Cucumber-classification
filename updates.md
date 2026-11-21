# Project updates

This file records changes made by Codex in DIP-project. Subsequent work should append an entry describing changed files, purpose, verification, results, pending work and any GitHub commit.

## 22 November 2026 — Rebuilt training and deployment, field readiness still unresolved

- Added corrected_data.py to rebuild a deterministic source-grouped 70/15/15 target split. Removed publisher-transformed tea variants from the new training inputs. Grouped known variants, exact duplicates and perceptually similar frames. Saved 909 views of 811 files in 778 groups, with 544/117/117 train/validation/test groups and zero group overlap. Plant and capture-session identities remain unavailable.
- Added fetch_field_data.py to download checksum-verified WMC-Leafset v4 Healthy and Leaf Miner damage images. Verified annotation IDs against the authors' paper. Mangalore cucumber is a proxy host. Existing tea and luffa proxy classes remain.
- Saved the newly downloaded WMC cache on the D drive after the C drive filled, with external_data/wmc4 linked to D:/BITS_Pilani/Academics/3rd year/sem1/MLEE/mlee-project/external_data/dip_wmc4_cache. The fetcher can recreate the cache on another machine.
- Added train_corrected.py. Fitted new five-class linear heads on frozen ImageNet MobileNetV2 features for raw and DIP images, and an RBF SVM on standardized HOG/LBP/HSV features. Training weights balance classes and source groups. Epoch, SVM C and DIP model selection use validation macro-F1.
- Shared EXIF handling, deterministic DIP processing and JPEG decoding between training and inference. Saved new corrected model files. MobileNet feature layers use pretrained weights with newly trained classifier heads.
- Measured held-out results on 135 views. Raw MobileNet accuracy 98.52%, macro-F1 0.9801. DIP MobileNet accuracy 95.56%, macro-F1 0.9436. DIP SVM accuracy 82.96%, macro-F1 0.7722. Validation selected DIP MobileNet for the demo.
- Replayed both deployed predictors on all 135 test views. Each reproduced its saved training-evaluation predictions. DIP MobileNet full-frame subset accuracy is 105/111, with Leaf Miner recall 2/4. The benchmark mixes proxy hosts and labelled symptom views.
- Added corrected confusion matrices, comparison plots, manifest, predictions, runtime versions, split audit and source hashes in results/corrected. Matched 111 full-frame test files measured 58,364,018 original bytes and 1,963,639 processed bytes, a 96.64% reduction. Resizing alone accounts for most measured storage savings.
- Added model selection to leaf_demo, restored direct script invocation for leaf_demo.py and evaluate_leaf_demo.py, and preserved five scores and stage panels. Added evaluate_field_set.py for CSV-labelled independent full-frame evaluation with source attribution and exact benchmark-duplicate rejection.
- Added rebuild_dip.ps1 and updated run_dip.ps1, README and the presentation guide. Declared requests and installed missing scikit-image in the existing environment.
- Checked the nine previously inspected Google photos using their provisional filename labels. The new default matches only 3/9 and predicts Whitefly on all six misses. These photos were excluded from training and tuning. The earlier checkpoints matched 4/9. A reliable presentation predictor remains unfinished.
- Pending work requires verified target-domain cucumber training photos, especially Red Mite and Yellow Mosaic, plus an independent labelled rehearsal set for all five classes. Further tuning against these nine diagnostic photos would compromise their role as a check.
- The submission report remains pending at the user's request.
- Final verification compiled all changed Python entry points, reproduced the manifest byte-for-byte, replayed both models' saved test predictions, ran direct-file SVM demo invocation and exercised the independent CSV evaluator on all nine diagnostic photos. Visually inspected the corrected confusion matrices. The scoped Git staging check passed. Existing unrelated edits and diagnostic photo files remain local.
- Prepared the repair implementation, corrected weights and measured outputs for publication to the existing GitHub repository. The chat records the resulting commit and push outcome.

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

## 22 November 2026 — Unlabelled demo input and five-class scores

- Added demo_input as the dedicated folder for presentation photos, with no class subfolders.
- Updated src/leaf_demo.py so running it without a photo argument processes all supported images directly in demo_input. Explicit photo paths still work.
- Added all five softmax percentage scores to terminal output and the DIP stage panel. The class with the highest score is the prediction.
- Kept the existing analyze return shape for evaluation callers, and added scores to its stage metadata.
- Added empty-folder guidance, unique-stem checking and per-image failure messages.
- Updated reports/PRESENTATION_GUIDE.md with the new command and the interpretation of scores.
- Verification uses the saved model on the existing healthy rehearsal sample and an unlabelled folder copy. See the chat for the completed checks.

## 22 November 2026 — Investigated poor Google-photo predictions

- Inspected the nine photos already present in demo_input and compared the current processed checkpoint with the raw checkpoint on the same originals.
- Verified checkpoint class order against the training ImageFolder. Mapping is correct.
- Both variants matched 4/9 provisional expected labels inferred from filenames. Independent source and label verification is pending.
- Saved per-image predictions, top softmax scores and segmentation decisions in results/metrics/external_photo_diagnosis.csv.
- Added reports/EXTERNAL_PHOTO_DIAGNOSIS.md with observed failures, dataset mismatch and requirements for a valid training and external validation iteration.
- The current benchmark does not establish whole-leaf demo readiness. No replacement checkpoint was selected.
- The user-provided photos and existing unrelated working-tree changes were retained locally.

## 22 November 2026 — Verified supplied dataset provenance and found source leakage

- Added the four supplied dataset links to reports/dataset_construction.md and documented web verification limits.
- Verified the local Roboflow YAML IDs and crop mapping. Inspected one annotated image from each selected pest class. The inspected Leaf Miner box surrounds an insect, which differs from the winding-damage demonstration photos.
- Counted local labels before project cleaning: Leaf Miner 20 source images / 77 boxes, Whitefly 67 source images / 327 boxes.
- Confirmed the tea publisher supplies transformed variants and checked the local source filenames.
- Matched all 300 current Red Mite images back to 126 source-photo IDs by exact file hashes. Versions of 55 source IDs cross split boundaries.
- Saved measured source overlap in results/metrics/tea_source_group_audit.json and updated the diagnostic and status documents to require a split rebuild and retraining.
- Existing checkpoints have not been retrained during this audit. Earlier claims of globally leakage-safe evaluation need the tea-group correction.
