# Contributing

The sole contributor is [urmeo](https://github.com/urmeo). Use short commit subjects
that name the work: `updated readme`, `updated results`, `updated pipeline`, or
`updated work` when a change spans several areas.

## Setup and checks

Use Python 3.9 or newer. CI tests Python 3.9–3.12 and uses Node 22 for browser tests.
Run these commands from the repository root:

```bash
python -m pip install -e ".[dev,research]" mypy pytest-cov
python -m pytest --cov=novavision
node --test tests/test_web_ui.js
ruff check novavision tests scripts server.py
ruff format --check novavision tests scripts server.py
mypy novavision
bash scripts/smoke_server.sh
```

Load model libraries only when needed. Add regression tests when behavior changes.

## App

```bash
python -m pip install -e ".[app,ml]"
NOVA_BACKEND=diffusers python server.py
```

Open `http://127.0.0.1:8000`. Models download on first use.
`NOVA_BACKEND=null` produces synthetic test images, but text analysis still loads
the emotion classifier. Hosted generation uses `NOVA_BACKEND=hf-api` and requires
`HF_TOKEN`.

For a public deployment, set `NOVA_API_TOKEN`, serve requests through TLS, and run
one worker:

```bash
NOVA_BACKEND=diffusers python -m gunicorn --workers 1 --threads 4 --timeout 300 --bind 127.0.0.1:8000 server:app
```

The default limits are 30 requests per minute per IP and two concurrent requests
per process. Change them with `NOVA_RATE_LIMIT` and `NOVA_MAX_CONCURRENCY`.
Request bodies are limited to 32 KiB. Set `CORS_ORIGINS` to the allowed origins.
Enable `NOVA_TRUST_PROXY=1` only behind a configured proxy.

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

Recompute summaries from the saved records:

```bash
python scripts/resummarize.py
python scripts/report.py --results outputs/generated/reanalysis/results.json
python scripts/correct_recovery.py
python scripts/power_analysis.py
python scripts/compare_probes.py outputs/results/probe_validation_scene.json outputs/results/probe_validation_scene_l14.json
python scripts/make_submission.py --system "system name"
```

These commands write to `outputs/generated/` and leave the saved results unchanged.
Submissions follow [the benchmark schema](docs/benchmark/submission.schema.json).

Generate new images or evaluate new data with these commands:

```bash
python -m pip install -e ".[ml,research]"
python -m novavision.experiments.run --backend diffusers --contents 2 --seeds 1 --width 256 --height 256 --save-images --out outputs/generated/pilot
python -m novavision.data.build_benchmark --n 100 --out outputs/generated/affectbench.csv
python -m novavision.experiments.run --backend diffusers --track text --benchmark outputs/generated/affectbench.csv --seeds 3 --out outputs/generated/text
python -m novavision.eval.validate_probe --hf-dataset xodhks/EmoSet118K --label-key emotion --n 400 --split train --seed 0 --out outputs/generated/probe_validation.json
```

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
