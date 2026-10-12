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
