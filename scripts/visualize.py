"""Create charts and tables from saved results, without running models or simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from render_results import SOURCES, checked_probe

from novavision.eval.metrics import confusion_matrix
from novavision.prompting import TIERS

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (*TIERS, "scene")
NAMES = {
    "raw": "Raw text",
    "naive": "Emotion label",
    "emotion": "Emotion cues",
    "affect": "Affect cues",
    "scene": "Scene diagnostic",
}
PROBES = {name: (first, second) for name, first, second in SOURCES}
INK, TEAL, CORAL = "#163858", "#168778", "#c76b47"


def read_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing saved result: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return report


def probability(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between 0 and 1")
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def checked_pilot(report: dict, labels: list[str]) -> list[str]:
    metrics, records = report["metrics"], report["records"]
    tiers = [tier for tier in CONDITIONS if tier in metrics]
    if not records or not all(tier in tiers for tier in ("raw", "emotion", "affect")):
        raise ValueError("Pilot needs saved records and raw, emotion, and affect metrics")
    chance = probability(metrics["chance"], "Pilot chance")
    if round(1 / len(labels), 4) != chance:
        raise ValueError("Pilot chance disagrees with the label count")
    known_labels = set(labels)
    for tier in tiers:
        subset = [record for record in records if record["tier"] == tier]
        if not subset or metrics[tier]["n"] != len(subset):
            raise ValueError(f"Pilot {tier}: saved record counts disagree")
        if any({row["intended"], row["predicted"]} - known_labels for row in subset):
            raise ValueError(f"Pilot {tier}: unknown emotion label")
        accuracy = probability(metrics[tier]["accuracy"], f"Pilot {tier} accuracy")
        observed = sum(row["intended"] == row["predicted"] for row in subset) / len(subset)
        if round(observed, 4) != accuracy:
            raise ValueError(f"Pilot {tier}: saved accuracy disagrees with records")
        bounds = metrics[tier]["accuracy_ci"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Pilot {tier}: expected two confidence bounds")
        low, high = [probability(value, f"Pilot {tier} confidence bound") for value in bounds]
        if not low <= accuracy <= high:
            raise ValueError(f"Pilot {tier}: confidence bounds do not contain the accuracy")
    return tiers


def checked_planning(report: dict) -> None:
    if report["method"] != "binomial_mixture_simulation" or report["protocol_power"] is not False:
        raise ValueError("Planning chart requires an illustrative binomial simulation")
    for field in ("chance", "probe_ceiling", "alpha", "target_power"):
        probability(report[field], f"Planning {field}")
    if not report["rows"]:
        raise ValueError("Planning chart needs saved simulation rows")
    for row in report["rows"]:
        for field in ("effect_strength", "recovery_accuracy", "power_at_planned_n"):
            probability(row[field], f"Planning {field}")
        size = row["n_for_80pct_power"]
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
            raise ValueError("Planning sample size must be a positive integer or null")
    if not any(row["n_for_80pct_power"] is not None for row in report["rows"]):
        raise ValueError("Planning chart has no reachable saved sample sizes")
    if report["target_power"] != 0.8:
        raise ValueError("Saved n_for_80pct_power rows require target_power=0.8")


def load_saved_results(source_dir: Path) -> dict:
    filenames = ["corrected_recovery.json", "power_analysis.json"]
    if (source_dir / "results.json").exists():
        filenames.append("results.json")
    filenames += [filename for pair in PROBES.values() for filename in pair]
    reports = {}
    for filename in filenames:
        path = source_dir / filename
        report = read_report(path)
        try:
            if filename.startswith("probe_validation"):
                checked_probe(report)
            elif filename == "power_analysis.json":
                checked_planning(report)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot chart {path}: {exc}") from exc
        reports[filename] = report
    for pair in PROBES.values():
        first, second = (reports[name] for name in pair)
        if first["gold"] != second["gold"] or first["labels"] != second["labels"]:
            raise ValueError("Paired probe reports have different gold labels or label orders")
    labels = reports[PROBES["Faces"][0]]["labels"]
    if reports[PROBES["Scenes"][0]]["labels"] != labels:
        raise ValueError("Face and scene validation reports have different emotion labels")
    if "results.json" in reports:
        try:
            checked_pilot(reports["results.json"], labels)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot chart {source_dir / 'results.json'}: {exc}") from exc
    return reports


def canvas(title: str, subtitle: str, note: str, *, square: bool = False):
    fig, ax = plt.subplots(figsize=(9.6, 7.5 if square else 6.1))
    fig.suptitle(title, x=0.07, y=0.96, ha="left", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.07, 0.88, subtitle, fontsize=11.5, color=INK)
    fig.text(0.07, 0.035, note, fontsize=10, color="#526371", linespacing=1.6)
    fig.subplots_adjust(left=0.22, right=0.93, bottom=0.22, top=0.79)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#bac4cc")
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    return fig, ax


def recovery_chart(pilot: dict):
    metrics = pilot["metrics"]
    tiers = [tier for tier in CONDITIONS if tier in metrics]
    fig, ax = canvas(
        "Pilot emotion recovery",
        "SD-Turbo · CLIP ViT-B/32 · 256 × 256 · CPU · one seed",
        "Points: saved recovery accuracy. Lines: saved item-bootstrap 95% confidence intervals.\n"
        "Scene diagnostic uses fixed emotion scenes and is separate from the paired tiers.",
    )
    for y, tier in enumerate(tiers):
        row = metrics[tier]
        accuracy, (low, high) = row["accuracy"], row["accuracy_ci"]
        color = CORAL if tier == "scene" else TEAL
        ax.errorbar(
            accuracy,
            y,
            xerr=[[accuracy - low], [high - accuracy]],
            fmt="o",
            color=color,
            capsize=5,
            markersize=9,
            linewidth=2,
        )
        ax.text(0.98, y, f"{accuracy:.1%} · n={row['n']}", va="center", ha="right", color=INK)
    ax.axvline(metrics["chance"], color=INK, ls="--", lw=1, zorder=0)
    ax.text(metrics["chance"] + 0.015, -0.64, f"Chance {metrics['chance']:.1%}", color=INK)
    if "scene" in tiers:
        ax.axhline(tiers.index("scene") - 0.5, color="#d4dce2", lw=1)
    ax.set_yticks(range(len(tiers)), [NAMES[tier] for tier in tiers])
    ax.set(xlim=(0, 1), ylim=(len(tiers) - 0.5, -0.85), xlabel="Recovered target emotion")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", color="#e8edf0", zorder=0)
    ax.set_axisbelow(True)
    return fig


def confusion_chart(pilot: dict, labels: list[str]):
    records = [row for row in pilot["records"] if row["tier"] == "emotion"]
    counts = confusion_matrix(
        [row["intended"] for row in records], [row["predicted"] for row in records], labels
    )
    majority = counts.sum(axis=0).argmax()
    fig, ax = canvas(
        "Where the pilot probe made errors",
        f"Emotion cues tier · CLIP ViT-B/32 · {len(records)} images · counts per cell",
        f"The probe predicted {labels[majority]} for "
        f"{counts[:, majority].sum()}/{len(records)} images.\n"
        "This is a probe measurement; it does not establish how people perceive these images.",
        square=True,
    )
    ax.imshow(counts, cmap="GnBu", vmin=0, vmax=max(1, counts.max()))
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(
        range(len(labels)), [f"{label} (n={n})" for label, n in zip(labels, counts.sum(axis=1))]
    )
    ax.set(xlabel="Predicted emotion", ylabel="Intended emotion")
    for i, row in enumerate(counts):
        for j, value in enumerate(row):
            text = str(value) if row.sum() else "n/a"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=12,
                color="white" if value > counts.max() * 0.6 else INK,
            )
    return fig


def validation_chart(reports: dict):
    fig, ax = canvas(
        "Saved probe validation",
        "Accuracy checked against the saved gold labels and predictions",
        "Image IDs and dataset revisions were not saved; image validation was not rerun.\n"
        "Scenes have no neutral examples. Transfer to generated images is unmeasured.",
    )
    y = np.arange(len(PROBES))
    for i, (name, color) in enumerate((("CLIP ViT-B/32", INK), ("CLIP ViT-L/14", TEAL))):
        values = [reports[files[i]]["accuracy"] for files in PROBES.values()]
        bars = ax.barh(y + (i - 0.5) * 0.29, values, height=0.25, color=color, label=name)
        ax.bar_label(bars, [f"{value:.2%}" for value in values], padding=7, color=INK)
    ax.set_yticks(y, [f"{name}\n(n={reports[files[0]]['n']})" for name, files in PROBES.items()])
    ax.set(xlim=(0, 0.65), ylim=(1.55, -0.65), xlabel="Saved prediction accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", color="#e8edf0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    return fig


def planning_chart(report: dict):
    rows = sorted(
        (row for row in report["rows"] if row["n_for_80pct_power"] is not None),
        key=lambda row: row["recovery_accuracy"],
    )
    fig, ax = canvas(
        "Illustrative sample-size curve",
        f"Saved binomial mixture simulation · {report['target_power']:.0%} target power",
        f"Independent trials · assumed {report['probe_ceiling']:.1%} validation ceiling transfers "
        "to generated images.\n"
        f"Chance {report['chance']:.1%} · α={report['alpha']:g} · binomial null; "
        "benchmark uses a permutation null.",
    )
    x = [row["recovery_accuracy"] for row in rows]
    y = [row["n_for_80pct_power"] for row in rows]
    ax.plot(x, y, "o-", color=TEAL, lw=2, markersize=8)
    for accuracy, size in zip(x, y):
        ax.annotate(
            str(size),
            (accuracy, size),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            color=INK,
        )
    ax.set(
        xlabel="Assumed recovery accuracy",
        ylabel="Images for 80% power",
        ylim=(0, max(y) * 1.22),
        xlim=(min(x) - 0.025, max(x) + 0.025),
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(color="#e8edf0")
    ax.set_axisbelow(True)
    ax.text(
        0.98,
        0.95,
        "Lines connect saved simulation points",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#526371",
    )
    return fig


def tables(reports: dict) -> str:
    lines = [
        "<!-- Generated by scripts/visualize.py from saved JSON; do not edit by hand. -->",
        "# Saved results",
        "",
        "These tables use the available saved JSON reports. "
        "No models, image generation, or simulations were rerun.",
        "",
    ]
    pilot = reports.get("results.json")
    if pilot is None:
        lines += [
            "Pilot records (`results.json`) are unavailable. "
            "Pilot charts and recovery tables were not regenerated.",
            "",
        ]
    else:
        metrics = pilot["metrics"]
        lines += [
            "## Pilot recovery",
            "",
            "SD-Turbo; CLIP ViT-B/32; CPU; 256 × 256 images; two neutral subjects; one seed. "
            "Intervals are the saved item-bootstrap 95% confidence intervals.",
            "",
            "| Condition | Images | Accuracy | 95% CI | Macro-F1 | Shuffled-label p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for tier in CONDITIONS:
            if tier not in metrics:
                continue
            row = metrics[tier]
            low, high = row["accuracy_ci"]
            lines.append(
                f"| `{tier}` | {row['n']} | {row['accuracy']:.2%} | {low:.2%}–{high:.2%} | "
                f"{row['macro_f1']:.4f} | {row['shuffled_control']['p_value']:.4f} |"
            )
        lines += [
            "",
            f"Uniform chance is {metrics['chance']:.2%} (1/7). "
            "The `scene` diagnostic uses fixed emotion scenes "
            "and is separate from the paired tiers. "
            "The pilot has no saved `naive` measurement. "
            "No tier has a shuffled-label p-value below .05.",
            "",
        ]
    lines += [
        "## Probe validation",
        "",
        "| Dataset | Model | Saved images | Accuracy | Macro-F1 | Source | Split | Seed |",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]
    for name, filenames in PROBES.items():
        for filename in filenames:
            row = reports[filename]
            lines.append(
                f"| {name} | `{row['model']}` | {row['n']} | {row['accuracy']:.2%} | "
                f"{row['macro_f1']:.4f} | `{row['source']}` | `{row['split']}` | {row['seed']} |"
            )
    lines += [
        "",
        "Gold label arrays agree within each model pair. Original image IDs and dataset "
        "revisions were not saved, so the arrays do not establish image identity. "
        "The scene sample has no neutral examples. "
        "Transfer from these samples to generated images is unmeasured.",
        "",
        "## Partial recovery correction",
        "",
    ]
    correction = reports["corrected_recovery.json"]
    partial, coverage = correction["estimable_class_comparison"], correction["coverage"]
    lines += [
        f"Correction is estimable for {coverage['estimable']}/{coverage['total']} classes. "
        "Full seven-class corrected recovery is unavailable. "
        f"The partial comparison covers {', '.join(partial['labels'])}.",
        "",
        "| Scope | Apparent recovery | Corrected recovery |",
        "|---|---:|---:|",
        f"| All seven classes | {correction['apparent_recovery']:.2%} | Unavailable |",
        f"| Five estimable classes | {partial['apparent_recovery']:.2%} | "
        f"{partial['corrected_recovery']:.2%} |",
        "",
        "Neutral has no validation support. The surprise probe is non-discriminating. "
        "The five-class estimate is not a correction of the full seven-class result.",
        "",
        "## Illustrative planning simulation",
        "",
    ]
    planning = reports["power_analysis.json"]
    lines += [
        f"Saved binomial mixture simulation: chance {planning['chance']:.2%}, "
        f"assumed probe ceiling {planning['probe_ceiling']:.2%}, α={planning['alpha']:g}, "
        f"target power {planning['target_power']:.0%}. "
        "It assumes independent trials and transfer of real-image validation accuracy to "
        "generated images. It uses a binomial null; the benchmark uses a fixed-label "
        "permutation null. These rows do not estimate the benchmark test's power.",
        "",
        "| Effect strength s | Assumed recovery | Images for 80% power | "
        f"Power at n={planning['planned_n']} |",
        "|---:|---:|---:|---:|",
    ]
    for row in planning["rows"]:
        size = row["n_for_80pct_power"]
        lines.append(
            f"| {row['effect_strength']:g} | {row['recovery_accuracy']:.2%} | "
            f"{size if size is not None else 'Unreachable'} | {row['power_at_planned_n']:.1%} |"
        )
    lines += ["", "The curve connects these saved points. It adds no fitted or simulated values."]
    return "\n".join(lines) + "\n"


def render(source_dir: Path, output_dir: Path, tables_path: Path) -> list[Path]:
    reports = load_saved_results(source_dir)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "novavision-results",
            "axes.labelsize": 11.5,
            "savefig.facecolor": "white",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in reports
    }
    description = json.dumps(
        {
            "sources_sha256": sources,
            "method": "Charts of saved JSON; no model inference or simulations rerun",
        },
        sort_keys=True,
    )
    pilot = reports.get("results.json")
    labels = reports[PROBES["Faces"][0]]["labels"]
    charts = []
    if pilot is not None:
        charts += [
            ("recovery", lambda: recovery_chart(pilot)),
            ("confusion", lambda: confusion_chart(pilot, labels)),
        ]
    charts += [
        ("probe-comparison", lambda: validation_chart(reports)),
        ("planning-curve", lambda: planning_chart(reports["power_analysis.json"])),
    ]
    paths = []
    for filename, make_chart in charts:
        fig = make_chart()
        for extension in ("svg", "png"):
            path = output_dir / f"{filename}.{extension}"
            metadata = {"Description": description}
            if extension == "svg":
                metadata["Date"] = None
            fig.savefig(path, dpi=180, metadata=metadata)
            if extension == "svg":
                lines = path.read_text(encoding="utf-8").splitlines()
                path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
            paths.append(path)
        plt.close(fig)
    tables_path.parent.mkdir(parents=True, exist_ok=True)
    tables_path.write_text(tables(reports), encoding="utf-8")
    manifest_path = output_dir / "sources.json"
    manifest_path.write_text(description + "\n", encoding="utf-8")
    return [*paths, tables_path, manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets/results")
    parser.add_argument("--tables", type=Path, default=ROOT / "docs/results-tables.md")
    args = parser.parse_args()
    try:
        paths = render(args.source_dir, args.output_dir, args.tables)
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Cannot create result visuals: {exc}\n")
    if not (args.source_dir / "results.json").exists():
        print("Pilot records unavailable; existing pilot charts were left unchanged.")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
