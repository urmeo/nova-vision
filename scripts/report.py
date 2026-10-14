"""Render saved experiment results as Markdown tables."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

from novavision.eval.metrics import cohens_h, holm_bonferroni
from novavision.experiments.run import ALL_CONDITIONS as CONDITIONS

CONDITIONING_TIERS = ("naive", "emotion", "affect")


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None and isfinite(value) else "n/a"


def _fmt_p(value: float | None) -> str:
    return "<0.001" if value is not None and 0 < value < 0.001 else _fmt(value)


def _rho(metrics: dict, key: str) -> str:
    """Format a correlation and its bootstrap interval when both bounds are finite."""
    rho = metrics.get(key)
    interval = metrics.get(f"{key}_ci")

    if (
        interval
        and len(interval) == 2
        and all(bound is not None and isfinite(bound) for bound in interval)
    ):
        return f"{_fmt(rho)} [{interval[0]:.2f}, {interval[1]:.2f}]"
    return _fmt(rho)


def _shuffled_note(metrics: dict) -> str:
    """Circularity check: does any conditioning tier beat the shuffled-label null?"""
    controls = {
        tier: (metrics.get(tier) or {}).get("shuffled_control") or {} for tier in CONDITIONING_TIERS
    }
    parts = [
        f"{tier} p={control['p_value']:.2f}"
        for tier, control in controls.items()
        if control.get("p_value") is not None
    ]
    if not parts:
        return ""

    null_mean = next(
        (
            control["null_mean"]
            for control in controls.values()
            if control.get("null_mean") is not None
        ),
        None,
    )
    base = f" (null mean {null_mean:.3f})" if null_mean is not None else ""
    return (
        "**Shuffled-label control:** one-sided permutation test of recovery against "
        f"randomly reassigned target emotions{base}: {', '.join(parts)}. Small p-values "
        "indicate higher recovery than with permuted labels."
    )


def metrics_table(metrics: dict) -> str:
    head = "| Condition | Recovery [95% CI] | Macro-F1 | Valence ρ†‡ | Arousal ρ†‡ | CLIP-T | n |"
    rule = "|" + "---|" * 7
    lines = [head, rule]
    for condition in CONDITIONS:
        row = metrics.get(condition)
        if not row:
            continue
        low, high = row["accuracy_ci"]

        accuracy = f"{_fmt(row['accuracy'])} [{_fmt(low)}, {_fmt(high)}]"
        lines.append(
            f"| {condition} | {accuracy} | {_fmt(row['macro_f1'])} | {_rho(row, 'valence_rho')} | "
            f"{_rho(row, 'arousal_rho')} | {_fmt(row['clip_t'])} | {row['n']} |"
        )
    chance = metrics.get("chance")
    base = metrics.get("raw", {}).get("majority_baseline")
    if chance is not None:
        note = f"\nUniform chance = {chance:.3f} (1/7)"
        if base is not None:
            note += f"; raw-condition majority-class baseline = {base:.3f}"
        note += ". Compare recovery with these baselines before interpreting it."
        lines.append(note)
    health = metrics.get("probe_health")
    if health:
        lines.append(
            f"\n**Probe health:** the probe used {health['distinct_labels']}/{health['n_labels']} "
            f"emotion labels across the conditioning tiers, predicting "
            f"'{health['majority_label']}' for {health['majority_rate']:.0%} of items; every "
            "recovery result should be interpreted alongside this label distribution."
        )
    shuffled = _shuffled_note(metrics)
    if shuffled:
        lines.append(f"\n{shuffled}")
    lines.append(
        "\n† For neutral-content runs, intended valence and arousal use per-emotion priors. "
        "These correlations measure separation between emotions rather than variation "
        "within an emotion."
    )
    lines.append(
        "\n‡ ρ is Spearman correlation. Brackets give bootstrap 95% confidence intervals "
        "when available. Wide intervals leave the correlation uncertain."
    )
    lines.append("\nCLIP-T is image–text cosine similarity; n is the number of images.")
    return "\n".join(lines)


def contrasts_table(contrasts: dict, metrics: dict | None = None) -> str:
    head = "| Contrast | Δ accuracy | Bootstrap 95% CI | Cohen's h | p | p test |"
    rule = "|" + "---|" * 6
    lines = [head, rule]
    metrics = metrics or {}
    for name, contrast in contrasts.items():
        interval = f"[{_fmt(contrast['ci_low'])}, {_fmt(contrast['ci_high'])}]"
        difference = contrast["mean_diff"]
        formatted_difference = (
            f"{difference:+.3f}" if difference is not None and isfinite(difference) else "n/a"
        )
        label = name.replace("_", " ")
        test = "exact McNemar" if contrast.get("p_method") == "mcnemar_exact" else "unrecorded"
        lines.append(
            f"| {label} | {formatted_difference} | {interval} | {_cohens_h(name, metrics)} | "
            f"{_fmt_p(contrast['p_value'])} | {test} |"
        )
    lines.append(
        "\nExact McNemar p-values are two-sided and use discordant pairs. "
        "Bootstrap intervals resample item pairs and can collapse to a point in small samples; "
        "they do not determine the exact test's p-value. "
        "P-values with an unrecorded test are retained as saved."
    )
    return "\n".join(lines)


def _cohens_h(contrast_name: str, metrics: dict) -> str:
    """Effect size for a tier-vs-tier contrast, from the two tiers' accuracies."""
    condition, _, baseline = contrast_name.partition("_vs_")
    accuracy = (metrics.get(condition) or {}).get("accuracy")
    baseline_accuracy = (metrics.get(baseline) or {}).get("accuracy")
    if accuracy is None or baseline_accuracy is None:
        return "n/a"
    return f"{cohens_h(accuracy, baseline_accuracy):+.3f}"


def _holm_note(metrics: dict) -> str:
    """Family-wise Holm correction over the conditioning tiers' shuffled-label p-values."""
    pvals = {}
    for tier in CONDITIONING_TIERS:
        control = (metrics.get(tier) or {}).get("shuffled_control") or {}
        if control.get("p_value") is not None:
            pvals[tier] = control["p_value"]
    if not pvals:
        return ""
    adjusted = holm_bonferroni(pvals)
    parts = [f"{cond} p_adj={adjusted[cond]['p_adjusted']:.2f}" for cond in pvals]
    survivors = [cond for cond in pvals if adjusted[cond]["reject"]]

    verdict = (
        "no tier rejects the shuffled-label null at α = 0.05"
        if not survivors
        else f"{', '.join(survivors)} reject the shuffled-label null at α = 0.05"
    )
    return (
        "**Holm correction:** adjusted shuffled-label p-values across the conditioning tiers: "
        f"{', '.join(parts)}; {verdict}."
    )


def render(results: dict) -> str:
    metrics = results["metrics"]
    holm = _holm_note(metrics)
    parts = [
        "<!-- generated by scripts/report.py; do not edit by hand -->",
        "**Table 1.** Emotion recovery by condition.",
        "",
        metrics_table(metrics),
        *(["", holm] if holm else []),
        "",
        "**Table 2.** Paired differences in recovery accuracy with bootstrap intervals.",
        "",
        contrasts_table(results["contrasts"], metrics),
        "",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render results as markdown tables")
    parser.add_argument("--results", default="outputs/generated/run/results.json")
    parser.add_argument("--out", default="outputs/generated/tables.md")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    table = render(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
