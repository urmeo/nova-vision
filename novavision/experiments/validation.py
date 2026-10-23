from __future__ import annotations

import math
from numbers import Integral, Real


def _integer_setting(value: int, name: str, *, minimum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or (minimum is not None and value < minimum)
    ):
        bound = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an integer{bound}")
    return int(value)


def _finite_setting(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and numeric")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite and numeric")
    return number


def _check_checkpoint_record(record, context: str = "Invalid checkpoint record") -> None:
    measurements = (
        "intended_valence",
        "intended_arousal",
        "recovered_valence",
        "recovered_arousal",
    )
    required = {
        "probe",
        "tier",
        "content",
        "index",
        "intended",
        "classified",
        "seed",
        "predicted",
        "image_pixel_sha256",
        *measurements,
    }
    if not isinstance(record, dict):
        raise ValueError(f"{context}: expected an object")
    missing = required - record.keys()
    if missing:
        raise ValueError(f"{context}: missing fields {', '.join(sorted(missing))}")
    for field in ("tier", "intended"):
        if not isinstance(record[field], str):
            raise ValueError(f"{context}: {field} must be a string")
    for field in ("index", "seed"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{context}: {field} must be a nonnegative integer")
    for field in measurements:
        if record[field] is not None:
            _finite_setting(record[field], f"{context}: {field}")
