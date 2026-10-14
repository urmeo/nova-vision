# Repository layout

| Path | Contents |
|---|---|
| `novavision/` | Python package: text analysis, prompts, generation, and evaluation. |
| `novavision/data/` | Text lexicon, content prompts, and dataset preparation. |
| `server.py` | Flask entry point for the local web interface. |
| `static/` | Browser interface and its text, styles, and scripts. |
| `scripts/` | Commands for summaries, charts, probe comparisons, submissions, and the static preview. |
| `notebooks/` | Notebook for checking saved probe and planning reports and running an optional new pilot. |
| `docs/` | Project documentation and the benchmark submission schema. |
| `docs/assets/` | Pipeline diagrams, screenshots, result charts, and videos with their viewing page and captions. |
| `tests/` | Python and browser-script tests, plus small test fixtures. |
| `outputs/results/` | Saved probe predictions, recovery correction, and planning simulation. |
| `outputs/figures/` | Figures and interface examples included with the project. |
| `outputs/generated/` | New runs and regenerated summaries; ignored by Git. |
| `.github/workflows/` | CI checks and GitHub Pages preview publishing. |

Run commands from the repository root. Keep saved records in `outputs/results/`;
write new experiments and temporary reports to `outputs/generated/`.

## Local use

Use Python 3.10 or later. Browser tests use Node.js 22 in CI.
The ML extra requires PyTorch 2.14.1 or later for native MPS interpolation.
Models download when missing from the local cache. Set `NOVA_BACKEND` for the app:

- `diffusers`: runs SD-Turbo locally. It selects CUDA, Apple MPS, then CPU.
  GPU runs use float16; CPU runs use float32 and can be slow.
- `null` (app default): makes test images. Text analysis still needs its classifier.
- `hf-api`: sends prompts to Hugging Face. It requires `HF_TOKEN` and a
  provider-supported model set through `NOVA_DIFFUSION_MODEL`.
  `NOVA_HF_TIMEOUT` sets its deadline in seconds (default: 120). A timed-out
  request stops its worker, releases the generation slot, and returns HTTP 504.

Hosted calls use a worker process. In scripts that call the Python API, put the
entry point under `if __name__ == "__main__":`. Worker shutdown can add up to two
seconds after the deadline. A provider may continue a job it already accepted.

## Experiments and reports

Run a small experiment and summarize its measurements:

```bash
python -m pip install -e ".[ml,research]"
python -m novavision.experiments.run --contents 1 --seeds 1 --width 256 --height 256 --save-images
python scripts/report.py
```

The run saves records, its manifest, and original images in `outputs/generated/run/`.
The summary goes to `outputs/generated/tables.md`.

Use `--diffusion-revision`, `--emotion-revision`, and `--probe-revision` to select
Hub commits, tags, or branches. `--clip-revision` also selects a CLIP probe revision.
Known defaults use pinned commits. Custom references resolve to full commit hashes
before loading; the manifest records those hashes and the requested references.
Pass full commit hashes for offline runs, or use an existing cached snapshot.
Local model directories are not supported by revision-pinned experiments.
`--emotion-model` and `--emotion-revision` apply to the text track.

Hosted inference cannot select a generator revision, so its diffusion revision
stays unknown and `--diffusion-revision` requires the local `diffusers` backend.
`--hf-timeout` sets the hosted experiment deadline in seconds (default: 120).

`python scripts/visualize.py` rebuilds the available saved charts and writes
`outputs/generated/results-tables.md`. Archived pilot charts and the
[published table](results-tables.md) stay intact; raw pilot records are no longer bundled.
