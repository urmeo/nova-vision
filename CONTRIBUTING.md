# Contributing

The sole contributor is [urme](https://github.com/urmeo). Use short lowercase
commit subjects that name the work: `updated readme`, `updated code`,
`updated results`, or `updated pipeline`. Use `updated work` only for unrelated
changes that have no shared purpose. Preserve existing commit history and keep
author and committer attribution on the `urmeo` account.

## Setup and checks

Use Python 3.9 or newer. CI tests Python 3.9–3.12 and uses Node 22 for browser tests.
Run these commands from the repository root:

```bash
python -m pip install -e ".[dev]" mypy pytest-cov
python -m pytest --cov=novavision
node --test tests/test_web_ui.js
ruff check novavision tests scripts server.py
ruff format --check novavision tests scripts server.py
mypy novavision
bash scripts/smoke_server.sh
```

These checks use synthetic inputs and mocks; they do not download model weights.
Install the `ml` extra for real inference and `research` for dataset downloads.
Add regression tests when behavior changes.

## App

```bash
python -m pip install -e ".[app,ml]"
NOVA_BACKEND=diffusers python server.py
```

Open `http://127.0.0.1:8000`. Models download on first use.
`NOVA_BACKEND=null` produces synthetic test images, but text analysis still loads
the emotion classifier. Hosted generation uses `NOVA_BACKEND=hf-api` and requires
`HF_TOKEN`.

For a public deployment, set `NOVA_API_TOKEN` to protect image generation, serve
requests through TLS, and run one worker. Text analysis remains unauthenticated.
The browser's API token field accepts the generation token.

```bash
NOVA_BACKEND=diffusers python -m gunicorn --workers 1 --threads 4 --timeout 300 --bind 127.0.0.1:8000 server:app
```

The API shares a default budget of 30 requests per minute per IP. At most two
generation requests run concurrently per process; extra requests receive HTTP
429. Change these limits with `NOVA_RATE_LIMIT` and `NOVA_MAX_CONCURRENCY`. Request
bodies are limited to 32 KiB and text to 3–2000 characters. Set `CORS_ORIGINS` only
when another origin needs access. Enable `NOVA_TRUST_PROXY=1` only behind a proxy
that replaces untrusted forwarding headers.

GitHub Pages publishes a static interface preview when its source files change on
`main`. The preview shows saved charts and disables analysis and generation. Build
it locally with `python scripts/build_pages.py`; the output is `build/pages/`.
The publishing workflow can also be run manually from GitHub Actions.

## Research commands

Regenerate the visual report from saved records. This writes charts and tables to
`docs/` without model downloads or new simulations:

```bash
python scripts/visualize.py
```

Recompute summaries and the partial correction from saved records, or create a
submission:

```bash
python scripts/resummarize.py
python scripts/report.py --results outputs/generated/reanalysis/results.json
python scripts/correct_recovery.py
python scripts/make_submission.py --system "system name"
```

These commands write to `outputs/generated/` and leave the saved results unchanged.
Submissions follow [the benchmark schema](docs/benchmark/submission.schema.json).

Run a new illustrative binomial simulation with `python scripts/power_analysis.py`.
Its output also goes to `outputs/generated/`. It does not estimate power for the
paired benchmark; see [the assumptions and saved curve](docs/results.md#illustrative-planning-curve).

Generate new images or evaluate new data with these commands:

```bash
python -m pip install -e ".[ml,research]"
python -m novavision.experiments.run --backend diffusers --contents 2 --seeds 1 --width 256 --height 256 --save-images --out outputs/generated/pilot
python -m novavision.data.build_benchmark --n 100 --out outputs/generated/affectbench.csv
python -m novavision.experiments.run --backend diffusers --track text --benchmark outputs/generated/affectbench.csv --seeds 3 --out outputs/generated/text
python -m novavision.eval.validate_probe --hf-dataset xodhks/EmoSet118K --label-key emotion --n 400 --split train --seed 0 --out outputs/generated/probe_validation.json
```

Run probe validation for both models on the same images, using `--clip-model` to
select the second model. Then compare the fresh reports:

```bash
python scripts/compare_probes.py path/to/probe-a.json path/to/probe-b.json
```

The comparison requires matching decoded-image fingerprints in the same order.
Matching labels alone do not verify image identity. The historical reports lack
these fingerprints and cannot support a verified paired comparison.

The model and dataset commands download their inputs. The first experiment generates
63 images: four paired tiers × two subjects × seven labels, plus seven separate
scene diagnostics. It includes the `naive` tier, which the saved pilot predates.

The default generator is SD-Turbo. The default image probe is CLIP B/32.
`--diffusion-model`, `--probe`, `--probe-model`, and `--clip-model` select alternatives.
Use `--resume` only with the original configuration. The saved pilot lacks complete
source and image provenance, so these commands produce a new run rather than
reproduce its original images.

## Data and ratings

- `novavision/data/content_bank.txt` contains 20 hand-authored neutral subjects.
  `tests/fixtures/affectbench_sample.csv` contains 56 synthetic sentences for tests.
  AffectBench is built from single-label GoEmotions examples mapped to seven emotion
  classes. By default, it uses the pinned test split, removes duplicate text and
  training-split overlaps, and records any classes with too few examples.
- The bundled affect lexicon contains demonstration values, not empirical norms.
  Download the Warriner norms with `python scripts/download_lexicon.py`, then set
  `NOVAVISION_LEXICON=outputs/generated/warriner.tsv`. You can also supply a licensed
  TSV with `word`, `valence`, and `arousal` columns. Valence ranges from -1 to 1;
  arousal ranges from 0 to 1. Negation reverses valence for matched words within the
  next two tokens. It does not change arousal or handle longer scopes.
- Save originals with `--save-images`, then run
  `python -m novavision.eval.human_study build --results outputs/generated/pilot`.
  The command verifies image hashes and writes images, a blank rating sheet, and
  class counts to `outputs/generated/human_study/`. Keep `key.csv` hidden from
  raters. Recruit at least three independent raters and ask them to leave a rating
  blank when they cannot identify an emotion. Analyze each rater separately with
  `python -m novavision.eval.human_study analyze --ratings ratings.csv --key outputs/generated/human_study/key.csv`.
  This reports agreement with the probe and accuracy against intended labels.
  Compute inter-rater agreement separately and report the sampled class counts.
- Preserve measurements, controls, and provenance. Keep fixture results separate
  from saved research records. The partial recovery correction covers five of seven
  classes. The power simulation assumes independent binomial trials and transfer
  of probe accuracy from real to generated images; it does not estimate power for
  the paired evaluation protocol.

## Private security reports

Use [a private advisory](https://github.com/urmeo/nova-vision/security/advisories/new).
Include the affected version, reproduction steps, and impact. Omit credentials and
personal data. Keep vulnerability details private until a fix is available.
