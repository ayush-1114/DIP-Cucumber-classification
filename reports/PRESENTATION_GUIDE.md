# Presenting the cucumber leaf project

## Before entering the room

Bring the presentation laptop, a phone or camera, a way to transfer a photo locally, the repository with its two saved checkpoints, and the working Python environment. For the full benchmark, bring the local data directory containing the manifest-listed raw images. On a fresh machine, use Python 3.12, create a virtual environment, install requirements.txt during setup, and test the following commands. The demo loads weights locally and uses CPU. Keep a sample photo and a rendered panel for rehearsal.

From the DIP-project directory in PowerShell:

~~~powershell
.\mlee-env\Scripts\python.exe -m src.leaf_demo "data/raw/test/Healthy/healthy_0009.jpg"
~~~

## When the examiner hands you a leaf

1. Take a sharp, evenly lit photo. Fill most of the frame with the leaf, include the leaf boundary, and keep visible symptoms in focus. Save it as a JPEG or PNG. For a phone using HEIC, export a JPEG first.
2. Run the same command with the new photo path. The example below assumes you have saved the photograph in the project directory.

~~~powershell
.\mlee-env\Scripts\python.exe -m src.leaf_demo ".\leaf.jpg"
Invoke-Item ".\results\demo\leaf\stages.png"
~~~

3. Read the five-way classifier's predicted class. Show the original and the padded view first. Explain how preserving aspect ratio keeps the shape of tunnels and lesions intact. Point out what foreground the colour mask retained, whether GrabCut refined it or used a fallback, and whether any leaf tissue was lost.
4. Show the masked CLAHE image that feeds the classifier. Point to any changes in the visibility of texture, then show the Otsu colour candidates and interior Canny edges as diagnostic views. Explain a false positive if visible, such as a vein responding to Canny or non-diseased tissue responding to Otsu. Compare input bytes with processed JPEG bytes. Smaller source files may grow after re-encoding.
5. If the result looks wrong, keep the original prediction visible and identify a plausible image-level cause. Some training classes contain close-ups or other host plants. If you take a second close-up, give it a new filename and show both outcomes. Each run writes processed.jpg and stages.png under results/demo/<photo stem>.

## DIP principles for the viva

| Stage | Operation and purpose | Evidence to show |
| --- | --- | --- |
| Geometry | Scale both axes by 224 divided by the longer side, then centre on a black square. A validity mask identifies padding. | Original dimensions, proportions and padded frame |
| Colour and segmentation | Blur in a 5 × 5 window, threshold OpenCV HSV (H 8–100, S > 35, V > 40), close/open with a 7 × 7 kernel, and fill retained contours. GrabCut refines foreground from colour seeds. Very low or high mask coverage triggers a fallback. | HSV prior, final mask and actual decision printed on screen |
| Contrast | Apply CLAHE on LAB lightness with clip limit 2 and an 8 × 8 tile grid. Local equalization reveals texture while clipping limits contrast amplification. | Masked and enhanced leaf |
| Lesion candidates | Choose an Otsu threshold on foreground LAB a* values by maximizing between-class variance. A weak or tiny sample yields an empty candidate map. | Highlighted candidate pixels and visually plausible false positives |
| Edges | Canny uses gradients, nonmaximum suppression and hysteresis at 50/150. Eroding the mask suppresses the artificial outline. | Interior edge map |
| Encoding and inference | Encode the 224 × 224 output at JPEG quality 95. Decode it, scale RGB to [0,1], apply ImageNet channel normalization and classify with MobileNetV2. | Byte counts, processed image and predicted class |

## Experiments and grading evidence

The rubric rewards before/after figures, analysis of differences, a baseline and discussion of errors. Use the live panel for DIP implementation and visual comparison, the 1,004-pair storage measurement for compression, and the raw CNN as the baseline. Explain that resizing, masking and encoding all affect file size. For the 151-image stage comparison at fixed JPEG quality, show Table 1 of the [technical report](DIP_TECHNICAL_REPORT.md).

For the presentation workflow benchmark, run the following once on the local dataset. The preprocessing decision uses only the pixels. Ground truth enters during scoring. It writes a per-image CSV and a classification report with a confusion matrix under results/metrics.

~~~powershell
.\mlee-env\Scripts\python.exe -m src.evaluate_leaf_demo
~~~

See [requirements, completed work, remaining work and extras](DIP_PROJECT_STATUS.md), [historical matched model comparisons](PJ_RESULTS_AND_DISCUSSION.md), and [the presentation workflow benchmark](../results/metrics/demo_classification_report.txt). For final rehearsal, record the actual class, prediction and mask quality for independently labelled photographs of whole cucumber leaves under presentation lighting. A rehearsal on the course test split establishes software operation and benchmark performance. Field performance requires independent photographs.

