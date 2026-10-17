# Pipeline figures

## Editable source and exports

| Figure | Editable source | Repository exports |
|---|---|---|
| Signal flow, desktop | [Figma, frame 14:2](https://www.figma.com/design/6lh7NsSYszXlyDrk6I1IaU?node-id=14-2) | [SVG](assets/pipeline.svg) · [PNG](assets/pipeline.png) |
| Pipeline overview, mobile | [Figma, frame 24:3](https://www.figma.com/design/6lh7NsSYszXlyDrk6I1IaU?node-id=24-3) | [SVG](assets/pipeline-mobile.svg) · [PNG](assets/pipeline-mobile.png) |
| Text analysis | [Figma, frame 5:2](https://www.figma.com/design/6lh7NsSYszXlyDrk6I1IaU?node-id=5-2) | [SVG](assets/analysis.svg) · [PNG](assets/analysis.png) |

Edit the Figma frames and export both formats when the method or labels change.
Keep the captions and README alt text consistent with the exports.
The [repository Figma files](../figma/) include a text-preserving SVG of the
signal-flow frame. Published SVGs use text outlines to preserve the layout without
installed fonts; PNGs use 1.5× scale. Keep the SVG title, description, and accessible
image role when replacing an export. Text equivalents appear below.
The README and website select the mobile figure on screens up to 900 pixels wide.

## Method shown

The app follows **Write text → Analyze text → Build prompt → Render image**.
Within text analysis, the classifier scores seven emotion labels and selects
the highest-scoring label. A word lexicon supplies valence and arousal values
for matched words. The analyzer blends those values with priors for the leading
label:

`estimate = w × word value + (1 − w) × label prior`, with
`w = min(matched content words / content words, 0.8)` by default.

The lexicon takes the arithmetic mean of matched word values. A two-word
negation window adjusts valence and stops at sentence punctuation, commas,
semicolons, colons, line breaks, clause dashes, and contrast markers such as
`but`. Quotes and grouping parentheses do not end the window.
Coverage weights that mean in the later blend. No matches
means the estimate comes from the label prior. The bundled word values are
demonstrations, not empirical norms. Valence ranges from −1 to 1; arousal ranges
from 0 to 1. These are text estimates, not measured responses to the generated
image.

For non-neutral text, `affect` adds mood wording, a valence-based palette, and
arousal-based lighting and composition. Neutral selects `raw`. The original text
and selected style enter the prompt directly, and both prompt paths retain fixed quality
terms. Seed and image dimensions enter the rendering backend. The app returns
an image and metadata. The default `null` backend creates synthetic test images;
local Diffusers and hosted inference are optional generation backends.

Experiments enter the same prompt builder and renderer with known targets and
paired prompt tiers. The four main conditions—`raw`, `naive`, `emotion`, and
`affect`—share a generator seed for each input, intended label, and seed index.
Other generation settings stay fixed. Experiments send generated images to a
probe and compare its estimates with the targets. The default probe is CLIP.
Probe predictions do not update prompts or regenerate images.

| Method or evidence | Implementation or record |
|---|---|
| Seven-label classification and affect blend | [Analyzer](../novavision/affect/analyzer.py) |
| Neutral selection and image rendering | [App pipeline](../novavision/pipeline.py) |
| Mood, palette, lighting, and tier definitions | [Prompt templates](../novavision/prompting.py) |
| Shared seeds, probe calls, and paired comparisons | [Experiment runner](../novavision/experiments/run.py) |
| Recovery, bootstrap intervals, exact paired binary tests, and label-permutation controls | [Metrics](../novavision/eval/metrics.py) |
| Saved pilot measurements | [Archived pilot table](results-tables.md#pilot-recovery) |

The content track crosses neutral subjects with supplied emotion labels and uses
fixed affect priors. The text track uses benchmark gold labels for prompt
conditioning; analyzer estimates supply affect targets. The saved pilot covers
only the content track.

New paired accuracy comparisons use a two-sided exact McNemar test on the
discordant pairs. Under equal correctness, either tier wins a discordant pair
with probability 0.5. Two wins and no losses give p=0.5. The
[exact test uses a binomial distribution](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).
Intervals still use a paired percentile bootstrap. Small samples can give
point intervals even when the exact test is not significant.

## Saved evidence

The [archived pilot table](results-tables.md#pilot-recovery) records the run settings,
recovery rates, and item-bootstrap 95% confidence intervals. Raw pilot records
are no longer bundled; the published charts and table remain.
The intervals describe uncertainty within this small pilot, not variation across
independent runs or human judgments.

Across the 42 conditioning-tier images, the probe predicted neutral 38 times (90.5%) and used
only two of seven labels. No measured condition exceeds the label-permutation
null at p < .05. The historical bootstrap test reported a paired difference
between `emotion` and `raw` of +7.14 percentage points (p=.2554);
`affect` and `emotion` have equal recovery
(p=1). These saved p-values have not been recomputed with the exact test because
the paired pilot records are no longer bundled. The pilot does not establish
emotion controllability.

`naive` has no saved measurement. The seven-image fixed-scene diagnostic is
separate from subject pairing.
Historical model pins, source commits, and original generated images are
incomplete, so exact image replay is unverified.

## Captions and text equivalents

**Signal flow, desktop.** English text branches into classification and word
scoring. The leading label supplies affect priors and selects the prompt tier.
Mean word values and coverage enter the affect blend. Original text and style
also enter the prompt directly. Neutral selects `raw`; other labels select
`affect`. Seed and dimensions enter the renderer, which returns an image and
run details. The separate experiment row pairs four prompt tiers with shared
seeds and settings, renders and probes each image, then compares estimates with
known targets. The dashed path retains intended targets for that comparison.
Probe estimates do not feed back into generation and are not human ratings.

**Pipeline overview, mobile.** The same method is condensed into four app steps
and three experiment steps. "Generate pairs" reuses prompt building and rendering.
Open the linked full-size signal-flow diagram to inspect the analysis branches.

**Text analysis.** Two sources feed the affect calculation: fixed priors for the
highest-scoring emotion label and mean values from matched words. Word coverage
sets the default blend weight, capped at 0.8. The outputs are valence (−1 to 1)
and arousal (0 to 1); no matches means the priors supply the values. A separate
annotation explains prompt selection: the leading label selects `raw` for
neutral or `affect` otherwise. Affect estimates determine palette and lighting
cues. Text scores do not verify a person's feelings.
