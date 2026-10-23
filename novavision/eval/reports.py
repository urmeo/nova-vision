from __future__ import annotations

import csv
import json
from pathlib import Path

from novavision.eval.metrics import accuracy, confusion_matrix, macro_f1, probability
from novavision.taxonomy import EMOTIONS

_PROBE_FIELDS = (
    "probe",
    "model",
    "source",
    "split",
    "seed",
    "revision",
    "n",
    "accuracy",
    "macro_f1",
    "label_order",
)
_CORRECTION_FIELDS = (
    "tier",
    "probe",
    "apparent_recovery",
    "corrected_recovery",
    "estimable",
    "total",
    "unestimable_labels",
    "scope",
    "estimable_labels",
    "partial_apparent_recovery",
    "partial_corrected_recovery",
)
_CLASS_FIELDS = ("apparent", "sensitivity", "specificity", "corrected", "status")
_ASSUMPTIONS = (
    "independent_trials",
    "non_conveyed_accuracy",
    "conveyed_accuracy",
    "mixture",
    "null",
    "validation_transfer",
)
_PLANNING_FIELDS = (
    "method",
    "protocol_power",
    "chance",
    "probe_ceiling",
    "alpha",
    "target_power",
    "planned_n",
    *(f"assumption_{name}" for name in _ASSUMPTIONS),
)
_EFFECT_FIELDS = (
    "effect_strength",
    "recovery_accuracy",
    "n_for_80pct_power",
    "power_at_planned_n",
)


def _integer(value: str, name: str, minimum: int = 0) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _probability(value: str, name: str, *, nullable: bool = False) -> float | None:
    if nullable and value == "":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    return probability(number, name)


def _boolean(value: str, name: str) -> bool:
    if value not in ("true", "false"):
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def _labels(value: str, name: str, *, complete: bool = False) -> list[str]:
    labels = value.split("|") if value else []
    if len(set(labels)) != len(labels) or set(labels) - set(EMOTIONS):
        raise ValueError(f"{name} contains duplicate or unknown labels")
    if complete and set(labels) != set(EMOTIONS):
        raise ValueError(f"{name} must contain all seven emotion labels")
    return labels


def _metadata(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict[str, str]:
    values = {name: rows[0][name] for name in fields}
    if any(row[name] != values[name] for row in rows for name in fields):
        raise ValueError("Repeated metadata disagrees between rows")
    return values


def _required(values: dict[str, str], *names: str) -> None:
    if any(not values[name].strip() for name in names):
        raise ValueError(f"Required fields must not be blank: {', '.join(names)}")


def _probe(rows: list[dict[str, str]]) -> dict:
    values = _metadata(rows, _PROBE_FIELDS)
    _required(values, "probe", "model", "source", "split")
    labels = _labels(values["label_order"], "label_order", complete=True)
    n = _integer(values["n"], "n", 1)
    if n != len(rows):
        raise ValueError("Saved prediction count disagrees with n")
    gold, predictions = [], []
    for index, row in enumerate(rows):
        if _integer(row["index"], "index") != index:
            raise ValueError("Observation indices must be consecutive and start at zero")
        if row["gold"] not in labels or row["prediction"] not in labels:
            raise ValueError("Observation contains an unknown emotion label")
        gold.append(row["gold"])
        predictions.append(row["prediction"])
    reported_accuracy = _probability(values["accuracy"], "accuracy")
    reported_macro_f1 = _probability(values["macro_f1"], "macro_f1")
    if reported_accuracy != round(accuracy(gold, predictions), 4):
        raise ValueError("Saved accuracy disagrees with predictions")
    if reported_macro_f1 != round(macro_f1(gold, predictions, labels), 4):
        raise ValueError("Saved macro-F1 disagrees with predictions")
    return {
        "probe": values["probe"],
        "n": n,
        "accuracy": reported_accuracy,
        "macro_f1": reported_macro_f1,
        "labels": labels,
        "confusion": confusion_matrix(gold, predictions, labels).tolist(),
        "gold": gold,
        "predictions": predictions,
        "model": values["model"],
        "source": values["source"],
        "split": values["split"],
        "seed": _integer(values["seed"], "seed"),
        "revision": values["revision"] or None,
    }


def _correction(rows: list[dict[str, str]]) -> dict:
    values = _metadata(rows, _CORRECTION_FIELDS)
    _required(values, "tier", "probe", "scope")
    labels = _labels("|".join(row["class"] for row in rows), "class", complete=True)
    estimable = _labels(values["estimable_labels"], "estimable_labels")
    unestimable = _labels(values["unestimable_labels"], "unestimable_labels")
    count = _integer(values["estimable"], "estimable")
    total = _integer(values["total"], "total", 1)
    if (
        count != len(estimable)
        or total != len(labels)
        or set(estimable) & set(unestimable)
        or set(estimable) | set(unestimable) != set(labels)
    ):
        raise ValueError("Class coverage disagrees with labels")
    expected_scope = "all_classes" if count == total else "partial"
    if values["scope"] != expected_scope:
        raise ValueError("Comparison scope disagrees with class coverage")
    per_class = {}
    statuses = {"estimable", "no_run_records", "no_validation_support", "non_discriminating_probe"}
    for label, row in zip(labels, rows, strict=True):
        status = row["status"]
        metrics = {
            name: _probability(row[name], name, nullable=True) for name in _CLASS_FIELDS[:-1]
        }
        if status not in statuses:
            raise ValueError("Unknown correction status")
        if (status == "estimable") != (label in estimable):
            raise ValueError("Class status disagrees with estimable labels")
        if (metrics["corrected"] is not None) != (status == "estimable"):
            raise ValueError("Corrected value disagrees with class status")
        if status == "estimable" and any(value is None for value in metrics.values()):
            raise ValueError(
                "Estimable classes require apparent recovery, sensitivity and specificity"
            )
        if status == "no_run_records" and metrics["apparent"] is not None:
            raise ValueError("Missing run records require unavailable apparent recovery")
        if status == "no_validation_support" and metrics["sensitivity"] is not None:
            raise ValueError("Missing validation support requires unavailable sensitivity")
        if status in ("no_validation_support", "non_discriminating_probe") and (
            metrics["apparent"] is None
            or (status == "non_discriminating_probe" and metrics["sensitivity"] is None)
        ):
            raise ValueError("Class status disagrees with available run or validation measurements")
        per_class[label] = {**metrics, "status": status}
    corrected = _probability(values["corrected_recovery"], "corrected_recovery", nullable=True)
    if count < total and corrected is not None:
        raise ValueError("Partial class coverage cannot supply an all-class corrected rate")
    return {
        "tier": values["tier"],
        "probe": values["probe"],
        "apparent_recovery": _probability(
            values["apparent_recovery"], "apparent_recovery", nullable=True
        ),
        "corrected_recovery": corrected,
        "coverage": {"estimable": count, "total": total, "unestimable_classes": unestimable},
        "estimable_class_comparison": {
            "scope": values["scope"],
            "labels": estimable,
            "apparent_recovery": _probability(
                values["partial_apparent_recovery"], "partial_apparent_recovery", nullable=True
            ),
            "corrected_recovery": _probability(
                values["partial_corrected_recovery"], "partial_corrected_recovery", nullable=True
            ),
        },
        "per_class": per_class,
    }


def _planning(rows: list[dict[str, str]]) -> dict:
    values = _metadata(rows, _PLANNING_FIELDS)
    _required(values, "method", *(f"assumption_{name}" for name in _ASSUMPTIONS[1:]))
    assumptions: dict[str, str | bool] = {
        name: values[f"assumption_{name}"] for name in _ASSUMPTIONS
    }
    assumptions["independent_trials"] = _boolean(
        values["assumption_independent_trials"], "independent_trials"
    )
    effects = []
    for row in rows:
        effect = {
            name: _probability(row[name], name)
            for name in _EFFECT_FIELDS
            if name != "n_for_80pct_power"
        }
        effect["n_for_80pct_power"] = (
            _integer(row["n_for_80pct_power"], "n_for_80pct_power", 1)
            if row["n_for_80pct_power"]
            else None
        )
        effects.append(effect)
    if len({row["effect_strength"] for row in effects}) != len(effects):
        raise ValueError("Simulation effect strengths must be unique")
    return {
        "method": values["method"],
        "protocol_power": _boolean(values["protocol_power"], "protocol_power"),
        "assumptions": assumptions,
        **{
            name: _probability(values[name], name)
            for name in ("chance", "probe_ceiling", "alpha", "target_power")
        },
        "planned_n": _integer(values["planned_n"], "planned_n", 1),
        "rows": effects,
    }


def _read_csv(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            fields = reader.fieldnames
            if not fields or len(set(fields)) != len(fields):
                raise ValueError("CSV needs unique column names")
            rows = list(reader)
        if not rows or any(None in row or None in row.values() for row in rows):
            raise ValueError("CSV needs complete rows")
        schemas = (
            (set(_PROBE_FIELDS) | {"index", "gold", "prediction"}, _probe),
            (set(_CORRECTION_FIELDS) | set(_CLASS_FIELDS) | {"class"}, _correction),
            (set(_PLANNING_FIELDS) | set(_EFFECT_FIELDS), _planning),
        )
        for columns, load in schemas:
            if set(fields) == columns:
                return load(rows)
        raise ValueError("Unrecognized report columns")
    except (ValueError, csv.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid CSV in {path}: {exc}") from exc


def read_report(path: Path) -> dict:
    try:
        if path.suffix.lower() == ".csv":
            return _read_csv(path)
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing saved result: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return report
