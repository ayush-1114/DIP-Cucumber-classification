# DIP project status and submission checklist

Updated 22 November 2026. This is a planning document for the DIP submission.

## Current repair status

Source-group training has been rebuilt and both corrected MobileNet and SVM predictors have been verified against saved test predictions. New measurements are in results/corrected. DIP MobileNet scored 129/135 views with macro-F1 0.9436, raw MobileNet 133/135 with macro-F1 0.9801, and DIP SVM 112/135 with macro-F1 0.7722. The MobileNet heads are newly fitted on frozen ImageNet features.

The new demo matches only 3/9 provisional Google-photo labels. A reliable unseen-cucumber demonstration remains unfinished. Verified cucumber training and independent evaluation photos are needed, particularly for Red Mite and Yellow Mosaic. Existing tea, luffa and Mangalore cucumber sources are proxies. Internal test scores cannot establish presentation readiness. See updates.md for the full repair log and reports/PRESENTATION_GUIDE.md for commands. The sections below record the earlier work and its historical measurements.

## What the course requires

The assigned task is a Python or MATLAB workflow that classifies cucumber leaf images into Whitefly, Red Mite, Yellow Mosaic, Leaf Miner and Healthy Leaf. It calls for resizing with aspect-ratio padding, compression, pixel rescaling, HSV colour processing, Sobel/Canny edges, CLAHE, Otsu thresholding and GrabCut. The workflow must enhance disease signatures, suppress non-leaf background, use a stratified 70/15/15 split and train a five-class classifier. The evaluation must show matched raw versus processed storage, feature evolution (original, colour masking, edge extraction and resized output) and a confusion matrix with per-class precision, recall and F1. [Assignment](../project-task.jpeg)

The grading guide also calls for labelled figures and tables, proper citations, before/after visuals, a baseline comparison and analysis of errors. Higher marks reward novelty or real-world relevance, a real dataset or application, and AI/ML or automation. The technical report needs an abstract, introduction, theory, methodology, implementation, results, discussion, conclusion, references and an appendix with scripts and execution instructions. After documented setup, the implementation should run end to end without intervention. The live demonstration adds a separate requirement from the presentation instructions: classify a previously unseen leaf on the day. [Grading and report guide](../guidelines.png)

## What is already done

- **Data and split.** Exact duplicate checking, deterministic downsampling and grouping of pest crops from the same source photo produced one shared manifest for 1,004 images (702 train, 151 validation and 151 test). Whitefly, Red Mite and Yellow Mosaic each contribute 210/45/45, Leaf Miner 43/10/10 and Healthy 29/6/6. [Construction notes](dataset_construction.md) · [Manifest](../data/split_manifest.csv)
- **DIP implementation.** OpenCV and NumPy resize and pad to 224 × 224, build an HSV foreground prior with morphology and hole filling, run adaptive GrabCut, and apply LAB luminance CLAHE. Otsu on LAB a* and Canny inside the leaf produce separate lesion and edge views. The classifier input is the masked and enhanced image. Canny covers the edge-detection requirement. Encoding follows the output filename extension and neural-network input is scaled and normalized through the PyTorch transforms. [Pipeline](../src/dip/pipeline.py) · [Training](../src/train_dl.py)
- **Classifiers and metrics.** HOG, LBP and HSV histograms feed SVM and Random Forest baselines. A five-class MobileNetV2 was trained on raw and processed variants. The local prediction files each match the 151 manifest test filename and class pairs. Raw MobileNetV2 scored 144/151 (95.36%, macro-F1 about 0.93), and processed MobileNetV2 scored 135/151 (89.40%, macro-F1 0.8588). The augmented SVM scored 88.74% with macro-F1 0.8017, and the augmented Random Forest scored 93.38% with macro-F1 0.8611. [Raw classification report](../results/metrics/dl_classification_report_raw.txt) · [Processed classification report](../results/metrics/dl_classification_report.txt) · [Classical results](../results/metrics/classical_ml_processed_summary.json)
- **Storage.** All 1,004 raw and processed image pairs were present locally. The manifest-listed files total 96,106,416 raw bytes and 18,834,991 processed bytes, or 96.11 and 18.83 decimal MB (80.40% less). The separate storage CSV uses recursively summed folders and binary units labelled MB. Its 90.49% result should not be used for the matched comparison. [Matched storage methodology](PJ_RESULTS_AND_DISCUSSION.md#storage-before-and-after-dip) · [Folder totals](../results/metrics/storage_metrics.csv)
- **Report material.** The existing MLEE report provides a starting point for theory, methodology, results, discussion, references and a contribution log. The repository also has confusion matrices and sample feature figures. The feature plot currently labels the already resized input as Original, and the Mask panel shows the final foreground mask. [Previous report](../MLEE_Final_Report.pdf) · [Example current figure](../results/figures/Yellow_Mosaic_evolution.png)

## Extra work to feature

- **Segmentation development.** Three iterations documented failure modes, including whole-leaf removal and lost pest or lesion detail, and introduced HSV seeds and GrabCut fallbacks. The documented 30-image check found zero black outputs after the revision. [Iteration notes](progress/02_dip_pipeline_fix.md)
- **Imbalance and rare-class analysis.** Training-only augmentation added 167 Leaf Miner and 181 Healthy samples. Random Forest Leaf Miner recall increased from 1/10 to 6/10, and overall macro-F1 went from about 0.746 to 0.861. A separate learning curve grouped Leaf Miner data by source photo. [Augmentation study](progress/03_classical_ml_augmentation.md) · [Learning curve](../results/figures/leaf_miner_learning_curve.png)
- **Ablation and hybrid modelling.** Seven Random Forest feature combinations were compared. LBP plus colour led validation with macro-F1 0.9063. The MobileNetV2 embedding SVM reached 95.36% test accuracy and macro-F1 0.8774, with Leaf Miner recall of 3/10 versus 8/10 for the original processed neural head. There is also a t-SNE illustration of HOG and learned features. [Analysis and methodology](PJ_RESULTS_AND_DISCUSSION.md)
- **Robustness and interpretability.** Grad-CAM artifacts compare raw and processed models. The saved synthetic stress test reports 34.44% raw and 55.63% processed accuracy under noise. The script adds Gaussian noise across the normalized input tensor, including leaf pixels. Describe it as a noise experiment and keep claims about unfamiliar backgrounds or field photos separate. [Raw Grad-CAM](../results/figures/gradcam_raw) · [Processed Grad-CAM](../results/figures/gradcam) · [Noise test](../src/background_swap.py)
- **Model storage.** Per-channel INT8 weight storage reduced the measured checkpoint from 9.144 MB to 2.654 MB (70.98%). Evaluation restored weights to FP32, so this measures checkpoint savings. [Quantization results](../results/storage_reduction/mobilenetv2_quantization.csv)

## Presentation and submission status

**Done.** The [live predictor](../src/leaf_demo.py) classifies an unlabelled photo on CPU. It generates a seven-stage panel, records the segmentation decision and compares source and processed file sizes. On all 151 held-out raw inputs it scored 87.42% accuracy and 0.8427 macro-F1. The [presentation guide](PRESENTATION_GUIDE.md) covers capture, execution, interpretation and viva questions.

**Additional DIP work.** A controlled experiment measures storage after resizing, CLAHE and masking on identical inputs at fixed JPEG quality. The [technical report](DIP_TECHNICAL_REPORT.md) includes theory, a flowchart, before/after visuals, measurements, baselines, classification and error analysis, references and an execution appendix. The [launcher](../run_dip.ps1) runs the demo and both evaluations after setup.

**Left before hand-in**

1. Rehearse on independently labelled whole cucumber leaves using the actual camera and presentation machine. Log expected class, prediction and mask quality. Prioritize Whitefly and Leaf Miner given the confusion observed on the held-out data.
2. Verify the training setup on a fresh installation, pin tested dependency versions and document source dataset acquisition. The presentation workflow uses the included checkpoint.
3. Add student/team identifiers and rehearsal observations, finalize dataset and method citations, export the approved submission format, and package the report, scripts, weights and documented data access.

## External-photo readiness check — 22 November 2026

A diagnostic check on nine user-provided whole-leaf photos matched 4/9 filename-derived expected classes with each existing checkpoint. The class mapping was verified. Source labels need verification. Whole-leaf presentation readiness requires further data and training work. See [external-photo diagnosis](EXTERNAL_PHOTO_DIAGNOSIS.md).


## Source audit correction — 22 November 2026

Recovered original tea filenames by hash matching. All 300 Red Mite examples mapped to 126 source photos. Transformed versions of 55 source IDs cross splits, including 34 shared between training and test. Rebuilding source-group splits and retraining is now a required correction. Earlier high Red Mite test scores are affected. Inspect all Roboflow pest annotations for compatibility with whole-leaf symptom classification before adding them to a revised training set. [Source audit](dataset_construction.md#source-audit-on-22-november-2026)
