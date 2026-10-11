"""Plot saved probe predictions without rerunning models or validating images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("Faces", "probe_validation.json", "probe_validation_l14.json"),
    ("Scenes", "probe_validation_scene.json", "probe_validation_scene_l14.json"),
)


def checked_probe(report):
    labels, gold, predicted = (report[k] for k in ("labels", "gold", "predictions"))
    if not gold or len(gold) != len(predicted) or report["n"] != len(gold):
        raise ValueError("Saved prediction counts disagree")
    if len(set(labels)) != len(labels) or (set(gold) | set(predicted)) - set(labels):
        raise ValueError("Saved labels disagree")
    index = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, prediction in zip(gold, predicted):
        counts[index[actual], index[prediction]] += 1
    score = sum(a == b for a, b in zip(gold, predicted)) / len(gold)
    if not np.array_equal(counts, report["confusion"]):
        raise ValueError("Saved confusion counts disagree with predictions")
    if round(score, 4) != report["accuracy"]:
        raise ValueError("Saved accuracy disagrees with predictions")
    return score, counts


def _canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(12, 7.6), dpi=100)
    fig.suptitle(title, fontsize=26, x=0.06, ha="left")
    fig.text(0.06, 0.88, subtitle, fontsize=16)
    fig.text(
        0.06,
        0.04,
        "Image IDs and revisions were not saved. No model rerun or human validation.",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.20, top=0.78)
    return fig, ax


def render(output_dir, font=None):
    if font is not None:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["font.size"] = 17
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports, sources = {}, {}
    for dataset, *filenames in SOURCES:
        pair = []
        for filename in filenames:
            path = ROOT / "outputs/results" / filename
            report = json.loads(path.read_text())
            score, counts = checked_probe(report)
            pair.append((report, score, counts))
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if pair[0][0]["gold"] != pair[1][0]["gold"]:
            raise ValueError("Probe reports do not share the same saved gold labels")
        reports[dataset] = pair

    def save(fig, filename, fields):
        metadata = {
            "sources_sha256": sources,
            "fields": fields,
            "method": "Recomputed from saved gold labels and predictions",
            "limit": "Original image IDs and revisions are absent; image validation was not rerun",
        }
        fig.savefig(output_dir / filename, metadata={"Description": json.dumps(metadata)})
        plt.close(fig)

    fig, ax = _canvas(
        "Recorded probe predictions", "Accuracy recomputed from saved gold labels and predictions"
    )
    y = np.arange(len(reports))
    for i, (model, color) in enumerate((("B/32", "#163858"), ("L/14", "#168778"))):
        values = [100 * pair[i][1] for pair in reports.values()]
        bars = ax.barh(y + (i - 0.5) * 0.34, values, height=0.30, color=color, label=model)
        ax.bar_label(bars, [f"{v:g}%" for v in values], padding=6)
    ax.set_yticks(y, [f"{name} (n={pair[0][0]['n']})" for name, pair in reports.items()])
    ax.invert_yaxis()
    ax.set(xlim=(0, 65), xlabel="Saved prediction accuracy (%)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "recorded-probe-accuracy.png", ["gold", "predictions", "n"])

    report, _, counts = reports["Scenes"][0]
    fig, ax = _canvas("Recorded scene errors", "CLIP B/32; 400 saved gold labels and predictions")
    ax.imshow(counts, cmap="GnBu", vmin=0, vmax=counts.max())
    ax.set_xticks(range(len(report["labels"])), report["labels"], rotation=30, ha="right")
    ax.set_yticks(range(len(report["labels"])), report["labels"])
    ax.set(xlabel="Predicted label", ylabel="Gold label")
    for i, row in enumerate(counts):
        for j, value in enumerate(row):
            text = "n/a" if row.sum() == 0 else str(value)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=13,
                color="white" if value > 35 else "#163858",
            )
    save(fig, "recorded-scene-confusion.png", ["gold", "predictions", "labels"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, help="optional path to a font file")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()
    render(args.output_dir, args.font)


if __name__ == "__main__":
    main()
