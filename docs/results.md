# Saved results

## Pilot recovery

![Pilot recovery accuracy and bootstrap intervals](assets/results/recovery.png)

The saved pilot used SD-Turbo, CLIP ViT-B/32, a CPU, 256 × 256 images, two neutral
subjects, and one seed. Each conditioning tier has 14 images; the separate scene
control has seven. The pilot predates `naive`, which has no saved measurement.

Chance agreement is 14.3% (1/7). No saved condition exceeds the shuffled-label
control at p < .05. Neither conditioning tier is significant after Holm
correction. The small sample and concentrated probe predictions limit what the
measurements say about perceived emotion.

![Emotion-tier confusion counts](assets/results/confusion.png)

Across conditioning tiers, the probe uses only two of seven labels and predicts
neutral for 90.5% of images. The pilot does not establish emotion controllability.

## Probe validation

![Probe accuracy on labelled face and scene samples](assets/results/probe-comparison.png)

The saved checks used `FastJobs/Visual_Emotional_Analysis` (faces, n=200) and
`xodhks/EmoSet118K` (scenes, n=400), with `split=train` and `seed=0`. The scene sample
has no neutral examples. These checks use training-split samples. Transfer to
generated images is unmeasured.

The saved label arrays give McNemar p=.040 for faces and p=.038 for scenes if
positions in each pair refer to the same images. Original image IDs and dataset
revisions are missing, so this pairing is unverified. The p-values are conditional
calculations, not verified evidence that one probe performs better.

New probe-validation reports include decoded-image fingerprints. The comparison
command requires matching fingerprints in the same order before computing a
paired test. Historical reports remain readable but cannot pass this check.

## Illustrative planning curve

![Binomial planning simulation from saved rows](assets/results/planning-curve.png)

This curve comes from a simulation, not a measured model run. It assumes
independent binomial trials and transfer of probe accuracy from real images to
generated images. It does not estimate power for the full paired evaluation
protocol. A run with 420 images per tier and a study with at least three
independent human raters remain future work.

The recovery correction covers five of seven classes for `emotion`: 10.0% apparent
recovery becomes 16.5% corrected recovery within that subset. A seven-class
estimate and correction uncertainty are unavailable.

## Records and regeneration

[Archived tables](results-tables.md) and [original figures](../outputs/figures/)
retain the published measurements. Raw pilot records are no longer bundled.
Historical source commits, complete model pins, and original generated images
are missing. Exact image replay is unverified. New runs write to
`outputs/generated/`.

Regenerate probe and planning charts from the remaining saved JSON files:

```bash
python -m pip install -e ".[research]"
python scripts/visualize.py
```

The script creates PNG and SVG charts in `docs/assets/results/` and replaces the
Markdown table report with tables for the available sources. Archived pilot
figures remain unchanged when raw records are absent. It does not download models
or overwrite saved records.
