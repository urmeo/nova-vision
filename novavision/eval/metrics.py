from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from numbers import Real

import numpy as np


def probability(value: object, name: str) -> float:
    error = f"{name} must be finite and between 0 and 1"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(error)
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(error) from exc
    if not 0 <= number <= 1:
        raise ValueError(error)
    return number


def _check(y_true: Sequence[object], y_pred: Sequence[object]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")


def _check_resampling(n: int, alpha: float = 0.05) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between 0 and 1")


def _check_labels(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> None:
    known = set(labels)
    if len(known) != len(labels):
        raise ValueError("labels must be unique")
    unknown = (set(y_true) | set(y_pred)) - known
    if unknown:
        raise ValueError(f"labels {sorted(unknown)} not in {sorted(known)}")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check(y_true, y_pred)
    if not y_true:
        return float("nan")
    correct = sum(t == p for t, p in zip(y_true, y_pred, strict=True))
    return correct / len(y_true)


def permutation_test(
    y_true: Sequence[str], y_pred: Sequence[str], *, n: int = 2000, seed: int = 0
) -> dict[str, float | list[float]]:
    _check(y_true, y_pred)
    _check_resampling(n)
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    if len(yt) < 2:
        nan = float("nan")
        return {"accuracy": nan, "null_mean": nan, "null_ci": [nan, nan], "p_value": nan}
    observed = float(np.mean(yt == yp))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(rng.permutation(yt) == yp)) for _ in range(n)])
    lo, hi = np.quantile(null, [0.025, 0.975])
    p = float((1 + np.sum(null >= observed)) / (n + 1))
    return {
        "accuracy": round(observed, 4),
        "null_mean": round(float(null.mean()), 4),
        "null_ci": [round(float(lo), 4), round(float(hi), 4)],
        "p_value": p,
    }


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    _check(y_true, y_pred)
    _check_labels(y_true, y_pred, labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[index[t], index[p]] += 1
    return matrix


def majority_baseline(y_true: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    counts = Counter(y_true)
    return max(counts.values()) / len(y_true)


def prediction_collapse(y_pred: Sequence[str]) -> dict:
    if not y_pred:
        return {"label": "", "rate": float("nan"), "distinct": 0}
    counts = Counter(y_pred)
    label, top = counts.most_common(1)[0]
    return {"label": label, "rate": top / len(y_pred), "distinct": len(counts)}


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    counts = confusion_matrix(y_true, y_pred, labels)
    support = counts.sum(axis=1)
    present = support > 0
    if not present.any():
        return float("nan")
    denominator = support + counts.sum(axis=0)
    return float(np.mean(2 * counts.diagonal()[present] / denominator[present]))


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> float:
    counts = confusion_matrix(a, b, labels)
    if not a:
        return float("nan")
    size = len(a)
    observed = counts.trace() / size
    expected = np.dot(counts.sum(axis=1) / size, counts.sum(axis=0) / size)
    if expected == 1.0:
        return float("nan")
    return float((observed - expected) / (1 - expected))


def _check_proportions(**values: float) -> None:
    for name, value in values.items():
        if value != value or value in (float("-inf"), float("inf")):
            continue
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")


def rogan_gladen(apparent: float, sensitivity: float, specificity: float) -> float:
    _check_proportions(apparent=apparent, sensitivity=sensitivity, specificity=specificity)
    denominator = sensitivity + specificity - 1.0

    if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(apparent):
        return float("nan")
    return float(np.clip((apparent + specificity - 1.0) / denominator, 0.0, 1.0))


def cohens_h(p1: float, p2: float) -> float:
    _check_proportions(p1=p1, p2=p2)
    if not (np.isfinite(p1) and np.isfinite(p2)):
        return float("nan")
    transformed = 2.0 * np.arcsin(np.sqrt(np.clip([p1, p2], 0.0, 1.0)))
    return float(transformed[0] - transformed[1])


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between 0 and 1")
    finite: dict[str, float] = {}
    for name, value in pvalues.items():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"p-value for {name} must be between 0 and 1")
        if not isinstance(value, Real):
            continue
        if not 0 <= value <= 1:
            if value != value or value in (float("-inf"), float("inf")):
                continue
            raise ValueError(f"p-value for {name} must be between 0 and 1")
        finite[name] = float(value)
    ordered = sorted(finite, key=finite.__getitem__)
    count = len(ordered)
    adjusted: dict[str, dict] = {}
    running = 0.0
    for i, name in enumerate(ordered):
        running = max(running, min(1.0, (count - i) * finite[name]))
        adjusted[name] = {"p_adjusted": round(running, 4), "reject": running <= alpha}
    for name in pvalues:
        if name not in adjusted:
            adjusted[name] = {"p_adjusted": None, "reject": False}
    return adjusted


def _numeric_vector(values: Sequence[float]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("values must be a one-dimensional numeric sequence") from exc
    if arr.ndim != 1:
        raise ValueError("values must be a one-dimensional numeric sequence")
    return arr


def _paired_vectors(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xa, ya = _numeric_vector(x), _numeric_vector(y)
    _check(x, y)
    return xa, ya


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = _paired_vectors(x, y)
    if len(xa) == 0:
        return float("nan")
    return float(np.mean(np.abs(xa - ya)))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = _paired_vectors(x, y)
    if len(xa) < 2 or xa.std() == 0 or ya.std() == 0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = _paired_vectors(x, y)
    if len(xa) < 2 or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        return float("nan")
    return pearson(_rank(xa).tolist(), _rank(ya).tolist())


def _rank(a: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    mean_ranks = counts.cumsum() - (counts + 1) / 2
    return mean_ranks[inverse]


def bootstrap_corr_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    method: str = "spearman",
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if method not in ("spearman", "pearson"):
        raise ValueError(f"unknown method '{method}', expected 'spearman' or 'pearson'")
    corr = spearman if method == "spearman" else pearson
    xa, ya = _paired_vectors(x, y)
    _check_resampling(n, alpha)
    if len(xa) < 3 or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(n):
        indices = rng.integers(0, len(xa), size=len(xa))
        correlation = corr(xa[indices].tolist(), ya[indices].tolist())
        if np.isfinite(correlation):
            correlations.append(correlation)
    if len(correlations) < 2:
        return float("nan"), float("nan")
    lo, hi = np.quantile(correlations, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_ci(
    values: Sequence[float], *, n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    _check_resampling(n, alpha)
    arr = _numeric_vector(values)
    if len(arr) < 2 or not np.isfinite(arr).all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.fromiter(
        (arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n)),
        dtype=float,
        count=n,
    )
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _binary_pair(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    if any(values.ndim != 1 or not np.isin(values, [0, 1]).all() for values in (da, db)):
        raise ValueError("a and b must be one-dimensional binary correctness arrays")
    if len(da) != len(db):
        raise ValueError("a and b must be the same length")
    return da, db


def mcnemar_exact(a: Sequence[float], b: Sequence[float]) -> dict[str, float | int]:
    da, db = _binary_pair(a, b)
    only_a = int(np.count_nonzero(da > db))
    only_b = int(np.count_nonzero(db > da))
    discordant = only_a + only_b
    coefficient = tail = 1
    for i in range(1, min(only_a, only_b) + 1):
        coefficient = coefficient * (discordant - i + 1) // i
        tail += coefficient
    p_value = min(1.0, (2 * tail) / (1 << discordant))
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p_value}


def paired_binary_test(
    a: Sequence[float], b: Sequence[float], *, n: int = 2000, seed: int = 0
) -> dict[str, float]:
    da, db = _binary_pair(a, b)
    _check_resampling(n)
    diff = da - db
    if len(diff) == 0:
        return {
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }
    lo, hi = bootstrap_ci(diff, n=n, seed=seed)
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(mcnemar_exact(a, b)["p_value"]),
    }


paired_bootstrap_test = paired_binary_test
