"Bias-correct recovery for the probe's measured error (Rogan-Gladen)."

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from novavision.eval.metrics import rogan_gladen
from novavision.taxonomy import EMOTIONS


def _sensitivity_specificity(confusion: list[list[int]]) -> dict[str, tuple[float, float]]:
    """Per-class (sensitivity, specificity) from a true x predicted count matrix."""
    cm = np.asarray(confusion, dtype=float)
    if cm.shape != (len(EMOTIONS), len(EMOTIONS)) or not np.isfinite(cm).all() or (cm < 0).any():
        raise ValueError("Validation confusion must be a finite nonnegative 7x7 matrix")
    out = {}
    for i, e in enumerate(EMOTIONS):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        out[e] = (sens, spec)
    return out


def _apparent_recovery(records: list[dict], tier: str) -> dict[str, float]:
    """Per-class apparent recovery rate for one tier."""
    out = {}
    for e in EMOTIONS:
        sub = [r for r in records if r["tier"] == tier and r["intended"] == e]
        out[e] = np.mean([r["predicted"] == e for r in sub]) if sub else float("nan")
    return out


def correct(results: dict, validation: dict, tier: str) -> dict:
    run_model = results.get("manifest", {}).get("config", {}).get("clip_model")
    validation_model = validation.get("model")
    if run_model and validation_model and run_model != validation_model:
        raise ValueError("Run and validation probe models differ")
    labels = list(validation.get("labels", EMOTIONS))
    if len(labels) != len(EMOTIONS) or set(labels) != set(EMOTIONS):
        raise ValueError("Validation labels must contain each canonical emotion exactly once")
    cm = np.asarray(validation["confusion"], dtype=float)

    _sensitivity_specificity(cm)
    order = [labels.index(e) for e in EMOTIONS]
    ss = _sensitivity_specificity(cm[np.ix_(order, order)])
    apparent = _apparent_recovery(results["records"], tier)
    per_class = {}
    corrected_vals = []
    estimable = []
    for e in EMOTIONS:
        a, (sens, spec) = apparent[e], ss[e]
        c = rogan_gladen(a, sens, spec) if a == a else float("nan")
        if not np.isfinite(a):
            status = "no_run_records"
        elif not np.isfinite(sens):
            status = "no_validation_support"
        elif not np.isfinite(c):
            status = "non_discriminating_probe"
        else:
            status = "estimable"
        per_class[e] = {
            "apparent": _round(a),
            "sensitivity": _round(sens),
            "specificity": _round(spec),
            "corrected": _round(c),
            "status": status,
        }
        if c == c:
            corrected_vals.append(c)
            estimable.append(e)
    complete = len(estimable) == len(EMOTIONS)
    apparent_macro = (
        float(np.mean(list(apparent.values())))
        if all(np.isfinite(a) for a in apparent.values())
        else np.nan
    )

    partial_corrected = float(np.mean(corrected_vals)) if corrected_vals else np.nan
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


def _round(x: float) -> float | None:
    return None if x != x else round(float(x), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rogan-Gladen recovery correction")
    parser.add_argument("--results", default="outputs/results/results.json")
    parser.add_argument("--probe-validation", default="outputs/results/probe_validation_scene.json")
    parser.add_argument("--tier", default="emotion")
    parser.add_argument("--out", default="outputs/generated/corrected_recovery.json")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    validation = json.loads(Path(args.probe_validation).read_text())
    report = correct(results, validation, args.tier)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
