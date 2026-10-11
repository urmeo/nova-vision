# NovaVision

Generate images from English text and emotion labels. Compare intended and recovered emotion in separate experiments.

![Text to image pipeline and separate experiment evaluation](docs/assets/pipeline.svg)

**Python · PyTorch · Transformers · Diffusers · NumPy · Pillow · Flask**

## Results

| Generator | Recovery probe | Device | Resolution | Subjects | Seeds |
|---|---|---|---|---:|---:|
| SD-Turbo | CLIP ViT-B/32 | CPU | 256 × 256 | 2 | 1 |

![Saved pilot recovery accuracy with bootstrap 95% confidence intervals](docs/assets/results/recovery.png)

The pilot does not establish emotion controllability. It has 14 images per conditioning tier and seven scene controls. The probe predicts neutral for 90.5% of conditioning-tier images. No condition exceeds the shuffled-label control at p < .05; `naive` is unmeasured.

![Recorded probe accuracy on labelled face and scene samples](docs/assets/results/probe-comparison.png)

These validation accuracies have not been shown to transfer to generated images. The scene sample has no neutral examples. Exact image replay is unverified because historical source and image provenance are incomplete.

[All plots, tables, and limitations](docs/results.md) · [Method and data](docs/method.md)

## Run

Python 3.9 or newer; CI checks 3.9–3.12.

```bash
git clone https://github.com/urmeo/nova-vision.git
cd nova-vision
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app,ml]"
NOVA_BACKEND=diffusers python server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Models download on first use. CPU generation can be slow. The default backend, `null`, creates test images; text analysis still needs its classifier. Hosted generation uses `NOVA_BACKEND=hf-api` and `HF_TOKEN` and sends prompts to the provider.

Check the code and regenerate the charts without model downloads:

```bash
python -m pip install -e ".[dev,research]"
python -m pytest
python scripts/visualize.py
```

## Repository

```text
novavision/  Analysis, prompts, generation, and evaluation
static/      Web interface; server.py serves it
scripts/     Research commands and chart generation
notebooks/   Saved-result checks and optional model runs
docs/        Method, visual results, and submission schema
tests/       Regression tests and synthetic fixtures
outputs/     Saved records, original figures, and new runs
```

[Interface preview](docs/assets/interface.png) · [Development](CONTRIBUTING.md) · [Notebook](notebooks/reproduce.ipynb) · [Folder guide](docs/structure.md)

Sole contributor: [urmeo](https://github.com/urmeo). Project code: [MIT](LICENSE). Model and dataset terms apply separately.
