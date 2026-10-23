from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from novavision.outputs import atomic_output

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _save_figure(fig: Figure, path: str | Path) -> None:
    path = Path(path)
    filetype = path.suffix[1:] or fig.canvas.get_default_filetype()
    if not path.suffix:
        path = path.with_suffix(f".{filetype}")
    with atomic_output(path) as stream:
        fig.savefig(stream, format=filetype, dpi=150)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


@contextmanager
def _plot(path: str | Path, figsize: tuple[float, float]) -> Iterator[Axes]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    try:
        yield ax
        fig.tight_layout()
        _save_figure(fig, path)
    finally:
        plt.close(fig)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Confusion matrix must be a nonempty finite nonnegative matrix")
    row_n = matrix.sum(axis=1, keepdims=True)
    if not np.isfinite(row_n).all():
        raise ValueError("Confusion matrix row totals must be finite")
    return np.divide(matrix, row_n, out=np.full_like(matrix, np.nan), where=row_n > 0)


def plot_confusion(
    matrix: np.ndarray, labels: Sequence[str], path: str | Path, title: str = ""
) -> None:
    norm = _row_normalize(matrix)
    if norm.shape != (len(labels), len(labels)) or len(set(labels)) != len(labels):
        raise ValueError("Confusion matrix must be square with one unique label per row")
    supported = np.isfinite(norm).any(axis=1)
    plt = _pyplot()

    cmap = plt.get_cmap("Blues").with_extremes(bad="#f0f0f0")
    with _plot(path, figsize=(5.5, 5)) as ax:
        ax.imshow(np.ma.masked_invalid(norm), cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        yticklabels = [
            label if present else f"{label} (n=0)"
            for label, present in zip(labels, supported, strict=True)
        ]
        ax.set_yticks(range(len(labels)), yticklabels)
        ax.set_xlabel("recovered")
        ax.set_ylabel("intended")
        if title:
            ax.set_title(title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                cell = f"{norm[i, j]:.2f}" if supported[i] else "n/a"
                ax.text(j, i, cell, ha="center", va="center", fontsize=8)


def plot_accuracy(
    tier_acc: dict[str, float],
    path: str | Path,
    chance: float | None = None,
    *,
    accuracy_ci: dict[str, list[float | None]] | None = None,
) -> None:
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in tier_acc.values()):
        raise ValueError("Recovery accuracies must be finite and between 0 and 1")
    if chance is not None and (not np.isfinite(chance) or not 0 <= chance <= 1):
        raise ValueError("Chance accuracy must be finite and between 0 and 1")
    from matplotlib.ticker import PercentFormatter

    tiers = list(tier_acc)
    with _plot(path, figsize=(5, 3.5)) as ax:
        ax.bar(tiers, [tier_acc[t] for t in tiers], color="#4c72b0")
        if accuracy_ci:
            for i, tier in enumerate(tiers):
                bounds = accuracy_ci.get(tier)
                if bounds is None:
                    continue
                if len(bounds) != 2:
                    raise ValueError(f"Accuracy interval for {tier!r} must have two bounds")
                low, high = bounds
                if low is None or high is None or not (np.isfinite(low) and np.isfinite(high)):
                    continue
                if not 0 <= low <= high <= 1:
                    raise ValueError(
                        f"Accuracy interval for {tier!r} must satisfy 0 <= low <= high <= 1"
                    )
                ax.vlines(i, low, high, color="#1e293b", linewidth=1.5)
                ax.hlines((low, high), i - 0.08, i + 0.08, color="#1e293b", linewidth=1.5)
            ax.set_title("Recovery accuracy · bootstrap 95% CI")
        ax.set_ylabel("Recovery accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1)
        for i, t in enumerate(tiers):
            ax.text(
                i, max(0.025, tier_acc[t] / 2), f"{tier_acc[t]:.1%}", ha="center", color="white"
            )
        if chance is not None:
            ax.axhline(chance, ls="--", lw=0.9, color="#888", label=f"Chance: {chance:.1%}")
            ax.legend(loc="upper right", frameon=False)


def plot_va_scatter(records, tier: str, path: str | Path) -> None:
    subset = [record for record in records if record["tier"] == tier]
    with _plot(path, figsize=(4.5, 4.5)) as ax:
        for dimension, color in (("valence", "#c44e52"), ("arousal", "#55a868")):
            ax.scatter(
                [record[f"intended_{dimension}"] for record in subset],
                [record[f"recovered_{dimension}"] for record in subset],
                alpha=0.6,
                label=dimension,
                color=color,
            )
        ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.8)
        ax.set_xlabel("intended")
        ax.set_ylabel("recovered")
        ax.set_title(f"valence/arousal ({tier})")
        ax.legend()
