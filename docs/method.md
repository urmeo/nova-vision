# Method

![NovaVision pipeline](assets/pipeline.svg)

## App: text to image

English text supplies two kinds of information: the subject to draw and estimates
of emotion and affect. The [analyzer](../novavision/affect/analyzer.py) scores
anger, disgust, fear, joy, neutral, sadness, and surprise with DistilRoBERTa through
Transformers. A separate [word lexicon](../novavision/affect/lexicon.py) estimates
valence and arousal from matched words. These are text estimates, not verified
feelings or human ratings.

| Step | Input → output |
|---|---|
| Classify text | Text → seven scores and the leading emotion |
| Estimate affect | Matched word values + leading emotion's priors → valence and arousal |
| Build prompt | Text + selected style + optional mood, palette, lighting, and composition cues → prompt |
| Render image | Prompt + seed + image dimensions → image |

![Classifier and word-lexicon branches, coverage-weighted affect blend, and automatic prompt selection](assets/analysis.svg)

The default blend applies to both valence and arousal:
`estimate = w × lexicon value + (1 − w) × emotion prior`, where
`w = min(matched content words / content words, 0.8)`.
With no matched words, coverage and `w` are zero, so the estimate is the emotion
prior. Valence ranges from −1 to 1; arousal ranges from 0 to 1. The bundled
lexicon contains demonstration values, not empirical norms. Experiments can
override `w` with `--force-coverage` to test the lexicon/prior blend.

The [app pipeline](../novavision/pipeline.py) chooses `raw` when the leading
emotion is neutral and `affect` otherwise. `raw` keeps the subject and style;
`affect` also adds mood wording, a valence-based palette, and arousal-based lighting
and composition. All prompts include the same quality preset. The
[prompt templates](../novavision/prompting.py) define these terms explicitly.

| Backend | Behavior and limits |
|---|---|
| `null` — app default | NumPy and Pillow create deterministic synthetic test images. This is not diffusion generation; text analysis still needs its classifier. |
| `diffusers` | PyTorch and Diffusers run locally. The default model is SD-Turbo with two steps. CPU generation can be slow. Turbo ignores negative prompts and uses zero guidance. |
| `hf-api` | Hugging Face hosts generation and requires `HF_TOKEN`. Prompts leave the local machine. Exact replay is not guaranteed, even with a seed. |

See the [backend implementations](../novavision/generation/). The static Pages
preview displays the interface and saved charts; analysis and generation require
the Flask server. The app does not run image-recovery evaluation or feed the
generated image back into the text analyzer.

## Experiments: image to measurement

The [runner](../novavision/experiments/run.py) defines targets before generation,
builds each prompt tier, generates images, reads them with a probe, and compares
recovered values with the targets. It supports two input tracks:

| Track | Inputs and intended emotion | Affect targets |
|---|---|---|
| Content | Up to 20 hand-authored neutral subjects, each crossed with seven intended emotions; no text classifier | Fixed priors for each intended emotion |
| Text | AffectBench sentences; the gold benchmark label conditions the prompt | Analyzer's lexicon/prior blend, whose priors use the classifier's prediction |

The text track records classifier accuracy separately. Its prompts use gold
labels rather than classifier predictions. The content track measures separation
between emotions; its fixed priors do not test affect variation within one
emotion. The text track permits varying affect targets, but has no saved pilot
results.

| Paired tier | Prompt content | Comparison |
|---|---|---|
| `raw` | Subject/text + style | No emotion conditioning |
| `naive` | Subject/text + bare emotion label + style | Minimal label baseline |
| `emotion` | Subject/text + mood cues + style | Mood wording |
| `affect` | Subject/text + mood, palette, lighting, composition, and style | Added affect cues |

For each input, intended emotion, and seed index, the four main tiers share the
same generator seed. Model, style, dimensions, and other run settings stay fixed.
Unlike the app, experiments apply every tier to neutral targets too. Prompt
presets are part of the evaluation conditions; changing them changes the
experiment.

| Diagnostic | What it checks |
|---|---|
| Fixed `scene`, content track | One predefined scene per emotion and seed, outside subject pairing |
| Wrong-label `shuffled`, text track | Mood wording for a deliberately wrong label, using that label's seed; scored against the wrong intended label |
| Label-permutation null, each condition | Reassign target labels while holding probe predictions fixed; test recovery against this null |

The wrong-label prompt diagnostic and label-permutation test are separate checks.
The scene and wrong-label conditions are not additional stages of the app.

| Image probe | What its outputs mean |
|---|---|
| CLIP, default ViT-B/32 | Compare image embeddings with seven sets of emotion descriptions. Separate text ladders estimate valence and arousal. CLIP-T measures image–text cosine similarity. |
| Supplied image classifier | Map model labels to the seven emotions. Affect values come from priors for the predicted label, not independent affect measurements. CLIP-T is unavailable. |

These [probe outputs](../novavision/eval/probes.py) are model estimates, not human
judgments. New [probe-validation reports](../novavision/eval/validate_probe.py)
record SHA-256 fingerprints of decoded RGB pixels and image dimensions. The [paired comparison
command](../scripts/compare_probes.py) requires matching fingerprints in the same
order; shared labels alone do not establish image identity. Evaluation records recovery accuracy, macro-F1, confusion matrices,
affect errors and correlations, bootstrap intervals, and paired tier differences.
Majority baselines and prediction-collapse counts help reveal an uninformative
probe. The [report](../scripts/report.py) also applies Holm correction to the
conditioning tiers' label-permutation p-values.

New runs write `results.json` and figures under the selected output directory.
The [manifest](../novavision/experiments/manifest.py) records configuration,
available model revisions, seeds, package/device details, and source/data hashes.
Each image record contains its pixel hash. `--save-images` retains measured
originals; `--resume` checks matching provenance before continuing a checkpoint.
Custom models may be unpinned, and seeds alone do not guarantee replay across
backends or environments. The [human-study harness](../novavision/eval/human_study.py)
can prepare blinded ratings from verified originals or verified reconstructions;
no human ratings are included in the saved results.

The saved pilot measured only `raw`, `emotion`, and `affect` at 14 images each,
plus seven fixed-scene images: SD-Turbo, CLIP ViT-B/32, CPU, 256 × 256, two neutral
subjects, and one seed. `naive` and the text track are unmeasured. Probe collapse,
small samples, and incomplete historical provenance limit these results; the
pilot does not establish emotion controllability. See [saved results](results.md).

## Data

AffectBench is built on demand from single-label GoEmotions examples mapped to
seven classes. By default, the builder samples the pinned test split, removes
normalized duplicates and text shared with the train split, and records class
shortages and a file hash. This filtering does not establish what other models
saw during training. The 56 synthetic sentences in `tests/fixtures/` are test data.

Model and dataset terms apply separately from the project's MIT license. Hosted
inference sends the generation prompt to its provider. Obtain consent for human
ratings and keep sensitive text out of hosted requests.

## References

- Radford et al. (2021). [CLIP](https://proceedings.mlr.press/v139/radford21a.html).
- Sauer et al. (2023). [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042).
- Demszky et al. (2020). [GoEmotions](https://aclanthology.org/2020.acl-main.372/).
- Yang et al. (2023). [EmoSet](https://arxiv.org/abs/2307.07961).
