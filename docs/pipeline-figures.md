# Pipeline figures

The figures explain how NovaVision builds an image prompt and how separate
experiments test emotion recovery. They describe the implemented method, not a
new model or demonstrated improvement in emotion control. Saved pilot evidence
is listed separately below.

## Editable source and exports

| Figure | Editable source | Repository exports |
|---|---|---|
| Integrated pipeline | [Figma, frame 20:2](https://www.figma.com/design/6lh7NsSYszXlyDrk6I1IaU?node-id=20-2) | [SVG](assets/pipeline.svg) · [PNG](assets/pipeline.png) |
| Text analysis | [Figma, frame 5:2](https://www.figma.com/design/6lh7NsSYszXlyDrk6I1IaU?node-id=5-2) | [SVG](assets/analysis.svg) · [PNG](assets/analysis.png) |

Edit the Figma frames and export both formats when the method or labels change.
Keep the captions and README alt text consistent with the exports. The diagrams
use labels and arrows as well as color to identify stages.
PNG exports use 1.5× scale. SVG text uses outlines to preserve the layout without
installed fonts. Keep the SVG title, description, and accessible image role when
replacing an export; the text equivalents below supply the full explanation.

## Method shown

The app follows **Write text → Analyze text → Build prompt → Render image**.
Within text analysis, the classifier scores seven emotion labels and selects
the highest-scoring label. A word lexicon supplies valence and arousal values
for matched words. The analyzer blends those values with priors for the leading
label:

`estimate = w × word value + (1 − w) × label prior`, with
`w = min(matched content words / content words, 0.8)` by default.

The lexicon takes the arithmetic mean of matched word values, adjusting valence
for nearby negation. Coverage weights that mean in the later blend. No matches
means the estimate comes from the label prior. The bundled word values are
demonstrations, not empirical norms. Valence ranges from −1 to 1; arousal ranges
from 0 to 1. These are text estimates, not measured responses to the generated
image.

The highlighted prompt stage shows the mechanism under study. For non-neutral
text, `affect` adds mood wording, a valence-based palette, and arousal-based
lighting and composition. Neutral selects `raw`. The original text and selected
style enter the prompt directly, and both prompt paths retain fixed quality
terms. Seed and image dimensions enter the rendering backend. The app returns
an image and metadata. The default `null` backend creates synthetic test images;
local Diffusers and hosted inference are optional generation backends.

Experiments enter the same prompt builder and renderer with known targets and
paired prompt tiers. The four main conditions—`raw`, `naive`, `emotion`, and
`affect`—share a generator seed for each input, intended label, and seed index.
Other generation settings stay fixed. The app returns its image and metadata;
the experiment branch sends generated images to a probe and compares the
estimates with the targets. The default probe is CLIP. The target line carries
known labels and affect targets directly to comparison. Probe predictions do
not update prompts or regenerate images.

| Method or evidence | Implementation or record |
|---|---|
| Seven-label classification and affect blend | [Analyzer](../novavision/affect/analyzer.py) |
| Neutral selection and image rendering | [App pipeline](../novavision/pipeline.py) |
| Mood, palette, lighting, and tier definitions | [Prompt templates](../novavision/prompting.py) |
| Shared seeds, probe calls, and paired comparisons | [Experiment runner](../novavision/experiments/run.py) |
| Recovery, bootstrap intervals, and permutation tests | [Metrics](../novavision/eval/metrics.py) |
| Saved pilot measurements | [Archived pilot table](results-tables.md#pilot-recovery) |

The content track crosses neutral subjects with supplied emotion labels and uses
fixed affect priors. The text track uses benchmark gold labels for prompt
conditioning; analyzer estimates supply affect targets. This distinction is
explained in the [method](method.md). The saved pilot covers only the content
track.

## Saved pilot evidence

The saved run used SD-Turbo, CLIP ViT-B/32, CPU inference, 256 × 256 images,
two neutral subjects, seven intended labels, and one seed. Recovery is the
fraction of probe predictions matching intended labels.

| Condition | Correct / images | Recovery | Item-bootstrap 95% CI |
|---|---:|---:|---:|
| `raw` | 2 / 14 | 14.29% | 0–35.71% |
| `emotion` | 3 / 14 | 21.43% | 0–42.86% |
| `affect` | 3 / 14 | 21.43% | 0–42.86% |

The values above come from the saved `metrics` entries. The intervals
resample per-image correctness values. They describe uncertainty within this
small pilot, not variation across independent runs or human judgments. Uniform
chance agreement is 14.29% (1/7).

Across these 42 images, the probe predicted neutral 38 times (90.5%) and used
only two of seven labels. No measured condition exceeds the label-permutation
null at p < .05. The saved paired difference between `emotion` and `raw` is
+7.14 percentage points (p=.2554); `affect` and `emotion` have equal recovery
(p=1). The pilot does not establish emotion controllability.

`naive` has no saved measurement. The seven-image fixed-scene diagnostic is
separate from subject pairing and is excluded from the table above.
Historical model pins, source commits, and original generated images are
incomplete, so exact image replay is unverified. See the [saved metric
tables](results-tables.md).

## Captions and text equivalents

**Integrated pipeline.** The app analyzes English text, builds a prompt, and
renders an image. The prompt receives the original text and style, the leading
emotion label, and blended valence and arousal. Neutral selects `raw`; other
labels select `affect`. Seed and dimensions enter the renderer. One output
branch returns the app's image and metadata. Experiments supply known targets
to paired prompt tiers, use the same prompt builder and renderer, then send
their images to a probe. The probe estimates emotion and affect. Comparison
uses these estimates and the known targets to save recovery and affect results,
records, and charts. Probe estimates are not human ratings and do not feed back
into generation.

**Text analysis.** Two sources feed the affect calculation: fixed priors for the
highest-scoring emotion label and mean values from matched words. Word coverage
sets the default blend weight, capped at 0.8. The outputs are valence (−1 to 1)
and arousal (0 to 1); no matches means the priors supply the values. A separate
annotation explains prompt selection: the leading label selects `raw` for
neutral or `affect` otherwise. Affect estimates determine palette and lighting
cues. Text scores do not verify a person's feelings.
