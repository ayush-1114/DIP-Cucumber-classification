# External-photo diagnostic check — 22 November 2026

After source-group retraining, the corrected DIP MobileNet matches 3/9 provisional filename labels. All six misses are predicted Whitefly. The diagnostic photos were excluded from training and tuning. New per-image outputs are in results/corrected/external_diagnostic.csv. The observations below describe the earlier checkpoints.

## Observed failure

On the nine user-provided photos in demo_input, the current DIP demo matched four filename-derived expected labels. The raw checkpoint also matched four. These are diagnostic counts on a convenience sample, with provisional labels taken from filenames. Photo provenance and disease labels need verification before this becomes a formal external evaluation.

Both models missed the two provisionally labelled Red Mite photos and two of three Leaf Miner photos. The processed model also missed one of two Yellow Mosaic photos. The raw model missed both Yellow Mosaic photos. Both matched the two provisionally healthy photos.

The processed model predicted Healthy Leaf on cuc2_redmite.jpg with a softmax score of 60.47%, and on cuc7_miner.jpg with 57.22%. Scores indicate relative preference among the five outputs and can be high on incorrect predictions.

The checkpoint class index order was checked against the processed training ImageFolder. Both use Healthy, Leaf_Miner, Red_Mite, Whitefly, Yellow_Mosaic.

[Per-image comparison](../results/metrics/external_photo_diagnosis.csv)

## Interpretation and next implementation work

Dataset provenance documents tight pest crops, tea-leaf Red Mite images and luffa Yellow Mosaic images. The external photos show full cucumber leaves with varied backgrounds, framing and symptom patterns. Failure in both raw and processed checkpoints supports a generalization problem. The present comparison alone cannot measure the contribution of source bias, crop mismatch, limited training coverage and learned features individually.

The live demo's measured 87.42% benchmark accuracy applies to the supplied grouped split. This nine-image check shows that whole-leaf demonstration readiness remains unresolved.

For the next training iteration:

1. Verify source URLs and intended labels for the external photos. Keep these nine as a diagnostic set and record their use in development.
2. Acquire and verify full cucumber leaf training examples for all five classes, especially Red Mite, Leaf Miner and Whitefly, with source-photo grouping and separate validation. Retain a new independently collected final test set.
3. Apply the same label-independent preprocessing to training, validation and inference. Check that masks retain symptoms before training.
4. Train and select variants using validation data. Compare raw, resize-only and full DIP inputs under matched training settings. Evaluate both whole-leaf external performance and DIP image quality.
5. Rehearse new independently labelled images. Report per-class recall and mask failures along with storage measurements.

The two existing checkpoints remain useful experimental baselines. Selecting a new model requires training and validation evidence on the intended whole-leaf inputs.


### Confirmed tea source overlap

Byte-hash matching mapped all 300 current Red Mite files back to the supplied tea dataset. They originate from 126 source-photo IDs. Transformed versions of 55 source IDs cross split boundaries, including 34 source IDs shared between training and test. The historical split protects Roboflow crop groups but permits source-related tea images across subsets. The old Red Mite test scores need replacement after a source-group rebuild and retraining. [Measured groups](../results/metrics/tea_source_group_audit.json)

For a corrected dataset, recover original source IDs before renaming, group all transformed versions with their original, and apply augmentation only to training data. Rebuild both raw and processed variants with the same group assignment and regenerate metrics. Select a whole-leaf dataset based on the actual symptom/class definition.
