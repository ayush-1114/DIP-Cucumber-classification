# Project updates

## Added repository navigation guide

- Inventoried the code, checkpoints, results, launchers and existing documents.
- Recorded the remaining corrected-model robustness/Grad-CAM reruns, fresh-install verification and field generalization limitations.
- Performed documentation changes only during this inventory. Directory restructuring and file deletion remain pending review.

## Clarified existing robustness evidence

- Rechecked src/background_swap.py, saved OOD metrics and the existing raw/processed Grad-CAM artifacts following the user's reminder.
- Corrected the status audit to credit the historical noise-robustness improvement, 55.63% processed versus 34.44% raw. The script perturbs the entire normalized input after preprocessing. Matched background replacement and corrected-model robustness results remain unverified.

## Rechecked assignment and rubric completion

- Read guidelines.png and project-task.jpeg directly and compared their requirements with the current pipeline and measured outputs.
- Added the current requirement recheck to reports/DIP_PROJECT_STATUS.md. Most requested operations and evaluation artifacts are implemented. Field background isolation, cucumber-specific generalization, final reporting and clean-install reproducibility remain incomplete.
- Documented that prepared training/validation examples can demonstrate functionality with disclosure, while unseen-image performance requires held-out evidence.
- No repository cleanup was performed because overall completion has not been established.

## Investigated segmentation and Whitefly overprediction

- Confirmed cuc4_yel has HSV foreground coverage 89.8%, triggering the >85% bypass and preserving its background. The old status incorrectly asserted that no background existed. Corrected both bypass messages to describe unresolved foreground/background.
- Added retained-area percentage to the normal predictions.csv so full-frame fallback is visible.
- Added src/diagnose_segmentation.py and ran four fixed preprocessing variants on all nine existing photos using the saved model. Saved all 36 comparisons and five class scores in results/corrected/segmentation_diagnostic.csv.
- For cuc4_yel, forced HSV masking changes Whitefly to Leaf Miner. Central GrabCut retains the whole frame and still predicts Whitefly. For cuc5_yel central GrabCut predicts Yellow Mosaic, while it changes the previously correct cuc6_miner prediction to Whitefly. These ablations do not establish a globally better segmentation method.
- Whitefly errors also occur with active segmentation on cuc7_miner and cuc8_miner. Source/host mismatch and forced five-class selection remain plausible contributors. Source bias is an inference, not a measured causal attribution.
- Model weights and preprocessing outputs remain the same in the deployed workflow. The status and CSV changes expose uncertainty. A segmentation algorithm change requires matched retraining and held-out evaluation before deployment.
- Normal demo rerun processed all nine images but could not refresh predictions.csv because Windows denied write access, consistent with the CSV being open in Excel. Added an explicit close-and-rerun message. Existing CSV remains accessible. The separate 36-row diagnostic CSV was generated successfully.

##  Changed demo export to a single CSV

- Replaced timestamped CSV exports with results/demo/predictions.csv, refreshed on each run. Running the default command includes all current demo_input photos in one file.
- Updated README and the presentation guide to reflect the requested single-file workflow.
- Ran the demo and verified predictions.csv contains exactly one row for each of the nine images in demo_input. Removed the superseded timestamped CSV generated in the previous change.

This file records changes made by Codex in DIP-project. Subsequent work should append an entry describing changed files, purpose, verification, results, pending work and any GitHub commit.

## Added demo CSV export

- Updated src/leaf_demo.py to save an Excel-readable CSV automatically on every demo run, for either model and both single-photo and folder inputs.
- Each row includes image name, predicted class, all five scores in percent, the selected model, segmentation status and processing errors. Failed images retain a row with their error.
- CSVs use unique timestamped filenames in the output directory to preserve earlier runs and allow an earlier CSV to remain open in Excel.
- Updated README and the presentation guide. Ran the normal direct-file demo on all nine existing photos. Verified nine CSV rows, five score columns, sums within 0.001 percentage points of 100, and every predicted class matching the largest score. Python compilation and scoped diff checks passed.

## Rebuilt training and deployment, field readiness still unresolved

- Added corrected_data.py to rebuild a deterministic source-grouped 70/15/15 target split. Removed publisher-transformed tea variants from the new training inputs. Grouped known variants, exact duplicates and perceptually similar frames. Saved 909 views of 811 files in 778 groups, with 544/117/117 train/validation/test groups and zero group overlap. Plant and capture-session identities remain unavailable.
- Added fetch_field_data.py to download checksum-verified WMC-Leafset v4 Healthy and Leaf Miner damage images. Verified annotation IDs against the authors' paper. Mangalore cucumber is a proxy host. Existing tea and luffa proxy classes remain.
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
- Published the repair implementation, corrected weights and measured outputs to the existing GitHub repository as commit 3d4a940. The first push was blocked by network access and automatic approval review reaching a usage limit. A later authorized retry succeeded.
- The user chose to proceed with the current datasets after reviewing the field-photo limitations. Further data acquisition is deferred. The demo runs with the validation-selected DIP MobileNet and retains the measured limitations in its documentation.

##  Retrospective record of initial setup and demo work

This entry records work completed in the preceding chat turns. The log was added after the user pointed out its omission.

### Repository setup — 5c0d35d

- Preserved existing Git history and created the private ayush-1114/DIP-Cucumber-classification repository.
- Added reports/DIP_PROJECT_STATUS.md with the assignment requirements, completed work, extra experiments and remaining work.
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

##  Unlabelled demo input and five-class scores

- Added demo_input as the dedicated folder for presentation photos, with no class subfolders.
- Updated src/leaf_demo.py so running it without a photo argument processes all supported images directly in demo_input. Explicit photo paths still work.
- Added all five softmax percentage scores to terminal output and the DIP stage panel. The class with the highest score is the prediction.
- Kept the existing analyze return shape for evaluation callers, and added scores to its stage metadata.
- Added empty-folder guidance, unique-stem checking and per-image failure messages.
- Updated reports/PRESENTATION_GUIDE.md with the new command and the interpretation of scores.
- Verification uses the saved model on the existing healthy rehearsal sample and an unlabelled folder copy. See the chat for the completed checks.

##  Investigated poor Google-photo predictions

- Inspected the nine photos already present in demo_input and compared the current processed checkpoint with the raw checkpoint on the same originals.
- Verified checkpoint class order against the training ImageFolder. Mapping is correct.
- Both variants matched 4/9 provisional expected labels inferred from filenames. Independent source and label verification is pending.
- Saved per-image predictions, top softmax scores and segmentation decisions in results/metrics/external_photo_diagnosis.csv.
- Added reports/EXTERNAL_PHOTO_DIAGNOSIS.md with observed failures, dataset mismatch and requirements for a valid training and external validation iteration.
- The current benchmark does not establish whole-leaf demo readiness. No replacement checkpoint was selected.
- The user-provided photos and existing unrelated working-tree changes were retained locally.

## Verified supplied dataset provenance and found source leakage

- Added the four supplied dataset links to reports/dataset_construction.md and documented web verification limits.
- Verified the local Roboflow YAML IDs and crop mapping. Inspected one annotated image from each selected pest class. The inspected Leaf Miner box surrounds an insect, which differs from the winding-damage demonstration photos.
- Counted local labels before project cleaning: Leaf Miner 20 source images / 77 boxes, Whitefly 67 source images / 327 boxes.
- Confirmed the tea publisher supplies transformed variants and checked the local source filenames.
- Matched all 300 current Red Mite images back to 126 source-photo IDs by exact file hashes. Versions of 55 source IDs cross split boundaries.
- Saved measured source overlap in results/metrics/tea_source_group_audit.json and updated the diagnostic and status documents to require a split rebuild and retraining.
- Existing checkpoints have not been retrained during this audit. Earlier claims of globally leakage-safe evaluation need the tea-group correction.

## Fresh setup on D drive

- Copied the current project into the user-specified D-drive destination. Created .fresh-env and redirected installer caches, TEMP, TMP, torch and matplotlib storage into .runtime on D.
- Linked source datasets already on D. Added verify_setup.ps1 for dependency, benchmark, storage, plot and demo checks. Installation and verification outcomes are pending.
- This directory is a project copy without Git history. The existing GitHub repository remains the publication destination.

- Fresh installation from requirements.txt succeeded. pip check reported no broken requirements. verify_setup.ps1 passed benchmark replay, storage, plotting and all nine demo inputs.
- Isolated retraining completed in .runtime/retrain-check. All three models reproduced the previous per-image test predictions, validation scores and test metrics exactly. The newly trained DIP model also passed deployment replay.
- Saved installer, verification, retraining and deployment logs plus installed_requirements.txt under .runtime. New environments, caches, temporary files, models and outputs from this check are on D.
- Fresh setup/retraining reproducibility is verified on this Windows machine. This result does not resolve the previously documented field-image limitations.

## Updated DIP ownership documentation

- Set AP as the sole DIP project owner. Reworded historical experiment descriptions in neutral technical language and removed outdated team-role instructions.
- Applied these documentation changes in the D-drive working copy.

## Rerunning extra experiments on corrected models

- Added extra_corrected.py for cached feature extraction, seven RF feature subsets, matched training-only augmentation, validation-selected embedding SVM, source-group learning curves and separately projected t-SNE plots.
- Added extra_neural.py for deterministically selected correct/error Grad-CAM examples, predicted/true-class maps, top-20-percent versus random deletion checks, per-channel INT8 weight storage and matched raw/DIP noise and brightness tests. Corruptions are applied before preprocessing.
- Runs are in progress. Outputs are written to results/corrected/extra_experiments. All caches and logs remain on D. Heatmap quality and measured benefits remain to be assessed.

- Completed all corrected extra experiments, including separate original post-normalization noise replication. Saved CSVs, four comparison figures, eight Grad-CAM panels, quantized storage checkpoint, embedding SVM and verification summary under results/corrected/extra_experiments.
- Verified cached baseline predictions for all three models and clean stress predictions against all 135 recorded test views. Inspected all eight CAM panels and all four comparison figures.
- Grad-CAM localization check used all 28 held-out Leaf Miner views. DIP crop enrichment 1.87 versus raw 1.33, but peak localization was 0/4 on full frames for both models. No claim of reliable full-leaf symptom localization is supported.
- Before-DIP noise mean accuracy 96.54% raw versus 90.62% DIP. Historical normalized-tensor noise mean accuracy 75.31% raw versus 67.65% DIP. The corrected models do not reproduce the old DIP noise advantage.
- INT8 weight storage saves 71.04%, with accuracy falling to 94.81%. Leaf Miner mean recall increases from 22.62% to 71.43% across 15 to 59 training source groups. Augmentation yields mixed results. Methods and limitations are in the experiment README.
- Added rerun_extras.ps1 and updated REPO_GUIDE.md. All work and outputs remain in the D-drive copy. The base demo models remain available; these experiments have not been published to GitHub.

- Connected the D-drive folder to the existing GitHub history and prepared the latest code, documentation and experiment outputs for publication from D.

## Completed editable DIP report
- Built a sectioned LaTeX report following the grading guide, with editable cover placeholders, supplied logos, original vector diagrams, IEEE-style references and corrected experiment evidence.
- Compiled a 14-page A4 report and rendered the pages for visual review. Verified that the Overleaf ZIP contains the exact compiled source and referenced figures.
- Saved DIP_Report_Overleaf.zip, DIP_Final_Report.pdf and DIP_Implementation_Appendix.zip in the D-drive checkout.


## DIP report submission format

- Reorganized the existing report under the course guideline headings and preserved its experiments, figures, tables and discussion.
- Added complete Python scripts, PowerShell launchers, dependencies and setup notes as listings in the PDF appendix. Refreshed the Overleaf source ZIP.

- Reduced the report appendix to the DIP pipeline and live demo scripts, linked the complete implementation on GitHub, and revised the wording to present the DIP study on its own. Rebuilt the Overleaf archive.

- Updated public documentation and launcher defaults for the standalone DIP repository. Removed the archived reference PDFs from the published tree.
