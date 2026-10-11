"Bias-correct recovery for the probe's measured error (Rogan-Gladen)."

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from novavision.eval.metrics import rogan_gladen
from novavision.taxonomy import EMOTIONS


def _sensitivity_specificity(
    confusion: list[list[int]], labels: Sequence[str] = EMOTIONS
) -> dict[str, tuple[float, float]]:
    """Per-class (sensitivity, specificity) from a true x predicted count matrix."""
    cm = np.asarray(confusion, dtype=float)
    if cm.shape != (len(EMOTIONS), len(EMOTIONS)) or not np.isfinite(cm).all() or (cm < 0).any():
        raise ValueError("Validation confusion must be a finite nonnegative 7x7 matrix")
    out = {}
    for i, emotion in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        out[emotion] = (sens, spec)
    return out


def _apparent_recovery(records: list[dict], tier: str) -> dict[str, float]:
    """Per-class apparent recovery rate for one tier."""
    out = {}
    for emotion in EMOTIONS:
        subset = [r for r in records if r["tier"] == tier and r["intended"] == emotion]
        out[emotion] = (
            float(np.mean([r["predicted"] == emotion for r in subset])) if subset else float("nan")
        )
    return out


def correct(results: dict, validation: dict, tier: str) -> dict:
    run_model = results.get("manifest", {}).get("config", {}).get("clip_model")
    validation_model = validation.get("model")
    if run_model and validation_model and run_model != validation_model:
        raise ValueError("Run and validation probe models differ")
    labels = list(validation.get("labels", EMOTIONS))
    if len(labels) != len(EMOTIONS) or set(labels) != set(EMOTIONS):
        raise ValueError("Validation labels must contain each canonical emotion exactly once")
    probe_rates = _sensitivity_specificity(validation["confusion"], labels)
    apparent = _apparent_recovery(results["records"], tier)
    per_class = {}
    corrected_values = []
    estimable = []
    for emotion in EMOTIONS:
        apparent_rate = apparent[emotion]
        sensitivity, specificity = probe_rates[emotion]
        corrected_rate = rogan_gladen(apparent_rate, sensitivity, specificity)
        if not np.isfinite(apparent_rate):
            status = "no_run_records"
        elif not np.isfinite(sensitivity):
            status = "no_validation_support"
        elif not np.isfinite(corrected_rate):
            status = "non_discriminating_probe"
        else:
            status = "estimable"
        per_class[emotion] = {
            "apparent": _round(apparent_rate),
            "sensitivity": _round(sensitivity),
            "specificity": _round(specificity),
            "corrected": _round(corrected_rate),
            "status": status,
        }
        if np.isfinite(corrected_rate):
            corrected_values.append(corrected_rate)
            estimable.append(emotion)
    complete = len(estimable) == len(EMOTIONS)
    apparent_macro = (
        float(np.mean(list(apparent.values())))
        if all(np.isfinite(a) for a in apparent.values())
        else np.nan
    )

    partial_corrected = float(np.mean(corrected_values)) if corrected_values else np.nan
    partial_apparent = float(np.mean([apparent[e] for e in estimable])) if estimable else np.nan
    return {
        "tier": tier,
        "probe": validation.get("model", validation.get("probe")),
        "apparent_recovery": _round(float(apparent_macro)),
        "corrected_recovery": _round(partial_corrected) if complete else None,
        "coverage": {
            "estimable": len(estimable),
            "total": len(EMOTIONS),
            "unestimable_classes": [e for e in EMOTIONS if e not in estimable],
        },
        "estimable_class_comparison": {
            "scope": "all_classes" if complete else "partial",
            "labels": estimable,
            "apparent_recovery": _round(partial_apparent),
            "corrected_recovery": _round(partial_corrected),
        },
        "per_class": per_class,
    }


def _round(value: float) -> float | None:
    return round(float(value), 4) if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Rogan-Gladen recovery correction")
    parser.add_argument("--results", default="outputs/results/results.json")
    parser.add_argument("--probe-validation", default="outputs/results/probe_validation_scene.json")
    parser.add_argument("--tier", default="emotion")
    parser.add_argument("--out", default="outputs/generated/corrected_recovery.json")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    validation = json.loads(Path(args.probe_validation).read_text(encoding="utf-8"))
    report = correct(results, validation, args.tier)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
