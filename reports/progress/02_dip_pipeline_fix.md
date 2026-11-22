# Progress 2: DIP Pipeline, Iteration 2 to Iteration 3

**DIP project maintainer:** AP. This note documents the inherited segmentation implementation.

**Date:** 16 Nov 2026

## Summary

Before: Iteration 2 seeded GrabCut with an Otsu mask and produced fully black or
inverted outputs on several classes. After: Iteration 3 seeds GrabCut with an HSV
colour mask, skips segmentation where there is no background, and produced no black
outputs on a 30-image test.

## Before (Iteration 2: Otsu seed, then GrabCut)

| Class | Failure |
|---|---|
| Yellow_Mosaic, Red_Mite | Output completely black: GrabCut removed the whole leaf |
| Whitefly | The white bug was treated as background; a black hole was punched where the bug was |
| Leaf_Miner | Only the dark hole was kept; the leaf was deleted |

Figures: `results/figures/Iteration_2_Otsu_Fail/*_evolution.png`

**Root causes**
- Otsu on greyscale separates *bright from dark*, not *leaf from background*. On a crop
  that is all leaf, it still splits something: the bug, or the hole.
- GrabCut was given only "probable" labels, so it was free to flip every pixel to
  background.
- The safety check only caught a completely black image, and a bare `except:` hid errors.

## After (Iteration 3: HSV seed, adaptive GrabCut)

| Change | Why |
|---|---|
| Seed GrabCut with an **HSV plant-colour mask** (hue brown-to-green, saturated) and fill holes | Colour separates leaf from a white/grey backdrop; filling keeps a white bug inside the leaf |
| **Skip GrabCut** when the mask covers > 85% (tight crop) or < 10% (unreliable), and always for Whitefly / Leaf_Miner crops | These crops have no background to remove |
| Give GrabCut **certain** foreground and background seeds, plus the padding as certain background | Stops it flipping everything to background |
| Fall back to the HSV mask if GrabCut keeps less than half of it | No near-empty outputs |
| **CLAHE after** segmentation | CLAHE changes the colours GrabCut relies on |
| **Otsu inside the leaf only**, on LAB a* | Now separates lesions (yellow/brown) from healthy green |
| Canny inside the eroded mask | The mask's own outline is not counted as an edge |

Test on 30 real images (6 per class):

| Class | Iteration 2 | Iteration 3 |
|---|---|---|
| Yellow_Mosaic | Black output | 5 segmented cleanly, 1 correctly kept whole |
| Red_Mite | Black output | 3 clean, 2 kept part of a dark background, 1 kept whole |
| Whitefly | Bug punched out | 4 correctly kept whole, 2 over-segmented (fixed by always skipping this class) |
| Leaf_Miner | Only the hole kept | All 6 correctly kept whole |
| Healthy | (worked) | All 6 segmented cleanly |
| **Black outputs** | **Several** | **0** |

Figures: `results/figures/*_evolution.png`

## Why it is better

- Every image now produces a usable leaf image for the models.
- The pipeline adapts to the image: full segmentation for leaves on a backdrop, none for
  tight crops.
- Otsu and Canny now describe the disease inside the leaf instead of the background.

## Limitations still open

- Some dark Red_Mite backgrounds are partly kept.
- The thresholds were hand-picked and checked on 30 images, not the full set.
- About a fifth of Red_Mite images are black-and-white, so colour segmentation cannot
  apply to them.
