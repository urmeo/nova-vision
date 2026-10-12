# Pipeline figures

The figures explain how NovaVision builds an image prompt and how separate
experiments test emotion recovery. They describe the implemented method and the
saved pilot. They do not demonstrate a new model or improved emotion control.

## Editable source and exports

| Figure | Editable source | Repository exports |
|---|---|---|
| Pipeline overview | [Figma, frame 2:2](https://www.figma.com/design/6lh7NsSYszXlyDrk6I1IaU?node-id=2-2) | [SVG](assets/pipeline.svg) · [PNG](assets/pipeline.png) |
| Text analysis | [Figma, frame 5:2](https://www.figma.com/design/6lh7NsSYszXlyDrk6I1IaU?node-id=5-2) | [SVG](assets/analysis.svg) · [PNG](assets/analysis.png) |

Edit the Figma frames and export both formats when the method or labels change.
Keep the captions, README alt text, and numeric evidence consistent with the
exports. The diagrams use labels and arrows as well as color to identify stages.
PNG exports use 1.5× scale. SVG text uses outlines to preserve the layout without
installed fonts. Keep the SVG title, description, and accessible image role when
replacing an export; the text equivalents below supply the full explanation.

## Method shown

The app follows four stages: **Write text → Estimate emotion → Shape the prompt
→ Render image**. Its classifier scores seven emotion labels. A word lexicon
supplies valence and arousal values for matched words. The analyzer blends those
values with priors for the highest-scoring label:

`estimate = w × word value + (1 − w) × label prior`, with
`w = min(matched content words / content words, 0.8)` by default.

The lexicon takes the arithmetic mean of matched word values, adjusting valence
for nearby negation. Coverage weights that mean in the later blend. No matches
means the estimate comes from the label prior. The bundled word values are
demonstrations, not empirical norms. Valence ranges from −1 to 1; arousal ranges
from 0 to 1. The overview calls these pleasantness and activation; they are the
same text estimates, not measured responses to the generated image.

The highlighted prompt stage shows the mechanism under study. For non-neutral
text, `affect` adds mood wording, a valence-based palette, and arousal-based
lighting and composition. Neutral selects `raw`. Both retain the original text,
style, and fixed quality terms. The default `null` backend creates synthetic test
images; local Diffusers and hosted inference are optional generation backends.

Separate validation begins with known targets, generates the prompt conditions,
reads each image with a probe, and compares predictions with targets. The four
main conditions—`raw`, `naive`, `emotion`, and `affect`—share a generator seed for
each input, intended label, and seed index. Other generation settings stay fixed.
The probe does not update prompts or regenerate images.

| Figure content | Implementation or record |
|---|---|
| Seven-label classification and affect blend | [Analyzer](../novavision/affect/analyzer.py) |
| Neutral selection and image rendering | [App pipeline](../novavision/pipeline.py) |
| Mood, palette, lighting, and tier definitions | [Prompt templates](../novavision/prompting.py) |
| Shared seeds, probe calls, and paired comparisons | [Experiment runner](../novavision/experiments/run.py) |
| Recovery, bootstrap intervals, and permutation tests | [Metrics](../novavision/eval/metrics.py) |
| Values in the evidence strip | [Saved results JSON](../outputs/results/results.json) |

The content track crosses neutral subjects with supplied emotion labels and uses
fixed affect priors. The text track uses benchmark gold labels for prompt
conditioning; analyzer estimates supply affect targets. This distinction is
explained in the [method](method.md). The saved pilot covers only the content
track.

## Evidence in the overview

The saved run used SD-Turbo, CLIP ViT-B/32, CPU inference, 256 × 256 images,
two neutral subjects, seven intended labels, and one seed. Recovery is the
fraction of probe predictions matching intended labels.

| Condition | Correct / images | Recovery | Item-bootstrap 95% CI |
|---|---:|---:|---:|
| `raw` | 2 / 14 | 14.29% | 0–35.71% |
| `emotion` | 3 / 14 | 21.43% | 0–42.86% |
| `affect` | 3 / 14 | 21.43% | 0–42.86% |

The displayed values come from the saved `metrics` entries. The intervals
resample per-image correctness values. They describe uncertainty within this
small pilot, not variation across independent runs or human judgments. Uniform
chance agreement is 14.29% (1/7).

Across these 42 images, the probe predicted neutral 38 times (90.5%) and used
only two of seven labels. No measured condition exceeds the label-permutation
null at p < .05. The saved paired difference between `emotion` and `raw` is
+7.14 percentage points (p=.2554); `affect` and `emotion` have equal recovery
(p=1). The pilot does not establish emotion controllability.

`naive` has no saved measurement. The seven-image fixed-scene diagnostic is
separate from subject pairing and is excluded from this evidence strip.
Historical model pins, source commits, and original generated images are
incomplete, so exact image replay is unverified. See the [full results and
limitations](results.md).

## Captions and text equivalents

**Pipeline overview.** English text and style enter a four-stage app flow:
write text, estimate emotion, shape the prompt, and render an image. Prompt
conditioning adds mood and visual cues for non-neutral labels. Below it, a
separate validation flow compares prompt conditions with known targets through
an image probe. The evidence strip reports a small saved content-track pilot,
including bootstrap intervals and concentrated neutral predictions. It does
not show human ratings or a feedback loop.

**Text analysis.** Two sources feed the affect calculation: fixed priors for the
highest-scoring emotion label and mean values from matched words. Word coverage
sets the default blend weight, capped at 0.8. The outputs are valence (−1 to 1)
and arousal (0 to 1); no matches means the priors supply the values. A separate
annotation explains prompt selection: the leading label selects `raw` for
neutral or `affect` otherwise. Affect estimates determine palette and lighting
cues. Text scores do not verify a person's feelings.

The presentation keeps one main flow, highlights its relevant mechanism, and
places limitations beside the evidence. This follows the figure guidance on
message, captions, and selective color in [Rougier et al., *Ten Simple Rules for
Better Figures*](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833).
The explicit conditions, uncertainty, and limits follow the reporting concerns
in the [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist).
These are presentation and reporting choices; their effect on reader
understanding has not been measured.
