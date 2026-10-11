# Repository layout

| Path | Contents |
|---|---|
| `novavision/` | Python package: text analysis, prompts, generation, and evaluation. |
| `novavision/data/` | Text lexicon, content prompts, and dataset preparation. |
| `server.py` | Flask entry point for the local web interface. |
| `static/` | Browser interface and its text, styles, and scripts. |
| `scripts/` | Commands for summaries, charts, probe comparisons, submissions, and the static preview. |
| `notebooks/` | Notebook for checking saved results and running an optional new pilot. |
| `docs/` | Project documentation and the benchmark submission schema. |
| `docs/assets/` | Pipeline diagrams, interface screenshots, and regenerated result charts. |
| `tests/` | Python and browser-script tests, plus small test fixtures. |
| `outputs/results/` | Saved pilot records and probe predictions. |
| `outputs/figures/` | Figures and interface examples included with the project. |
| `outputs/generated/` | New runs and regenerated summaries; ignored by Git. |
| `.github/workflows/` | CI checks and GitHub Pages preview publishing. |

Run commands from the repository root. Keep saved records in `outputs/results/`;
write new experiments and temporary reports to `outputs/generated/`.
