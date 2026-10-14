"Classification and correlation metrics (numpy only)."

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def _check(y_true: Sequence[object], y_pred: Sequence[object]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")


def _check_labels(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> None:
    known = set(labels)
    unknown = (set(y_true) | set(y_pred)) - known
    if unknown:
        raise ValueError(f"labels {sorted(unknown)} not in {sorted(known)}")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check(y_true, y_pred)
    if not y_true:
        return float("nan")
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def permutation_test(
    y_true: Sequence[str], y_pred: Sequence[str], *, n: int = 2000, seed: int = 0
) -> dict[str, float | list[float]]:
    "Shuffled-label control for circularity: is recovery above random targets?"
    _check(y_true, y_pred)
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
        "p_value": round(p, 4),
    }


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    _check(y_true, y_pred)
    _check_labels(y_true, y_pred, labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    return matrix


def majority_baseline(y_true: Sequence[str]) -> float:
    "Accuracy of the degenerate classifier that always predicts one label."
    if not y_true:
        return float("nan")
    counts = Counter(y_true)
    return max(counts.values()) / len(y_true)


def prediction_collapse(y_pred: Sequence[str]) -> dict:
    "How concentrated a probe's predictions are, a degeneracy diagnostic."
    if not y_pred:
        return {"label": "", "rate": float("nan"), "distinct": 0}
    counts = Counter(y_pred)
    label, top = counts.most_common(1)[0]
    return {"label": label, "rate": top / len(y_pred), "distinct": len(counts)}


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Macro-F1 averaged over the labels actually present in y_true."""
    _check(y_true, y_pred)
    _check_labels(y_true, y_pred, labels)
    observed = set(y_true)
    present = [label for label in labels if label in observed]
    if not present:
        return float("nan")
    scores = []
    for label in present:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> float:
    """Chance-corrected agreement between two label sets (e.g. human vs probe)."""
    _check(a, b)
    _check_labels(a, b, labels)
    if not a:
        return float("nan")
    size = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / size
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum((counts_a[label] / size) * (counts_b[label] / size) for label in labels)
    if expected == 1.0:
        return float("nan")
    return float((observed - expected) / (1 - expected))


def rogan_gladen(apparent: float, sensitivity: float, specificity: float) -> float:
    "Prevalence corrected for an imperfect test (Rogan & Gladen, 1978)."
    denominator = sensitivity + specificity - 1.0

    if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(apparent):
        return float("nan")
    return float(np.clip((apparent + specificity - 1.0) / denominator, 0.0, 1.0))


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions (arcsine-transformed difference)."""
    if not (np.isfinite(p1) and np.isfinite(p2)):
        return float("nan")
    transformed = 2.0 * np.arcsin(np.sqrt(np.clip([p1, p2], 0.0, 1.0)))
    return float(transformed[0] - transformed[1])


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    "Family-wise correction over a set of named p-values (Holm, 1979)."
    finite = {
        name: value
        for name, value in pvalues.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }
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


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    """Mean absolute error."""
    _check(x, y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) == 0:
        return float("nan")
    return float(np.mean(np.abs(xa - ya)))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    _check(x, y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2 or xa.std() == 0 or ya.std() == 0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation."""
    _check(x, y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2 or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        return float("nan")
    return pearson(_rank(xa).tolist(), _rank(ya).tolist())


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)

    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def bootstrap_corr_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    method: str = "spearman",
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    "Percentile bootstrap CI for a correlation, resampling item pairs."
    if method not in ("spearman", "pearson"):
        raise ValueError(f"unknown method '{method}', expected 'spearman' or 'pearson'")
    corr = spearman if method == "spearman" else pearson
    _check(x, y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
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
    """Percentile bootstrap CI for the mean, resampling over items."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
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
    "Two-sided exact McNemar test, conditional on discordant binary pairs."
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
    "Exact McNemar p-value and a paired percentile-bootstrap interval for mean a - b."
    da, db = _binary_pair(a, b)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
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


# Compatibility name: binary inputs, exact p-value, bootstrap interval.
paired_bootstrap_test = paired_binary_test
