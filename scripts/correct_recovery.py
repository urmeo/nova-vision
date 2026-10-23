from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from numbers import Real
from pathlib import Path

import numpy as np

from novavision.eval.metrics import rogan_gladen
from novavision.eval.reports import read_report
from novavision.outputs import validate_output_paths, write_output
from novavision.prompting import FLOORS, TIERS
from novavision.taxonomy import EMOTIONS

_CONDITIONS = (*TIERS, *FLOORS, "shuffled")


def _sensitivity_specificity(
    confusion: object, labels: Sequence[str] = EMOTIONS
) -> dict[str, tuple[float, float]]:
    labels_error = "Validation labels must contain each canonical emotion exactly once"
    try:
        labels = list(labels)
    except TypeError:
        raise ValueError(labels_error) from None
    if (
        len(labels) != len(EMOTIONS)
        or any(not isinstance(label, str) or label not in EMOTIONS for label in labels)
        or len(set(labels)) != len(EMOTIONS)
    ):
        raise ValueError(labels_error)
    counts_error = "Validation confusion must be a finite nonnegative 7x7 matrix of integer counts"
    try:
        counts = np.asarray(confusion, dtype=object)
    except (TypeError, ValueError):
        raise ValueError(counts_error) from None
    if counts.shape != (len(EMOTIONS), len(EMOTIONS)):
        raise ValueError(counts_error)
    cm = np.empty(counts.shape, dtype=float)
    for index, value in np.ndenumerate(counts):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(counts_error)
        try:
            count = float(value)
        except OverflowError:
            raise ValueError(counts_error) from None
        if not np.isfinite(count) or value < 0 or value % 1 != 0:
            raise ValueError(counts_error)
        cm[index] = count
    with np.errstate(over="ignore"):
        total = cm.sum()
    if not np.isfinite(total):
        raise ValueError(counts_error)
    out = {}
    for i, emotion in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        out[emotion] = (sens, spec)
    return out


def _apparent_recovery(records: list[dict], tier: str) -> dict[str, float]:
    intended: Counter[str] = Counter()
    recovered: Counter[str] = Counter()
    for record in records:
        if record["tier"] == tier:
            label = record["intended"]
            intended[label] += 1
            recovered[label] += record["predicted"] == label
    return {
        emotion: recovered[emotion] / intended[emotion] if intended[emotion] else float("nan")
        for emotion in EMOTIONS
    }


def correct(results: dict, validation: dict, tier: str) -> dict:
    if not isinstance(results, dict) or not isinstance(validation, dict):
        raise ValueError("Run and validation reports must be objects")
    if not isinstance(tier, str) or tier not in _CONDITIONS:
        raise ValueError(f"Unknown correction tier {tier!r}; expected one of {_CONDITIONS}")
    records = results.get("records")
    if not isinstance(records, list):
        raise ValueError("Run records must be a list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object")
        if not isinstance(record.get("tier"), str) or record["tier"] not in _CONDITIONS:
            raise ValueError(f"Record {index} has an unknown tier")
        for field in ("intended", "predicted"):
            label = record.get(field)
            if not isinstance(label, str) or label not in EMOTIONS:
                raise ValueError(f"Record {index} {field} must be a canonical emotion label")
    manifest = results.get("manifest", {})
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be an object")
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Run manifest config must be an object")
    run_model = config.get("probe_model")
    validation_model = validation.get("model")
    if run_model and validation_model and run_model != validation_model:
        raise ValueError("Run and validation probe models differ")
    revisions = manifest.get("model_revisions", {})
    if not isinstance(revisions, dict):
        raise ValueError("Run manifest model_revisions must be an object")
    run_revision = revisions.get("probe") or config.get("probe_revision")
    validation_revision = validation.get("model_revision")
    if run_revision and validation_revision and run_revision != validation_revision:
        raise ValueError("Run and validation probe revisions differ")
    probe_rates = _sensitivity_specificity(
        validation.get("confusion"), validation.get("labels", EMOTIONS)
    )
    apparent = _apparent_recovery(records, tier)
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
    parser.add_argument("--results", default="outputs/generated/run/results.json")
    parser.add_argument("--probe-validation", required=True)
    parser.add_argument("--tier", default="emotion")
    parser.add_argument("--out", default="outputs/generated/corrected_recovery.json")
    args = parser.parse_args()

    results_path = Path(args.results)
    validation_path = Path(args.probe_validation)
    out = Path(args.out)
    try:
        validate_output_paths([out], [results_path, validation_path])
        results = json.loads(results_path.read_text(encoding="utf-8"))
        validation = read_report(validation_path)
        report = correct(results, validation, args.tier)
    except ValueError as exc:
        parser.error(str(exc))

    payload = json.dumps(report, indent=2, allow_nan=False).encode("utf-8")
    write_output(out, payload)
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
