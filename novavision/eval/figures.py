"""Plots for the affect-recovery experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    "pyplot on a non-interactive backend."
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    "Row-normalized confusion; a class absent from y_true becomes NaN, not 0."
    row_n = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_n.clip(min=1), dtype=float)
    norm[row_n.ravel() == 0, :] = np.nan
    return norm


def plot_confusion(matrix: np.ndarray, labels, path: str | Path, title: str = "") -> None:
    plt = _pyplot()

    row_n = matrix.sum(axis=1, keepdims=True)
    norm = _row_normalize(matrix)
    cmap = plt.get_cmap("Blues").with_extremes(bad="#f0f0f0")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(np.ma.masked_invalid(norm), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    yticklabels = [
        lab if n else f"{lab} (n=0)" for lab, n in zip(labels, row_n.ravel(), strict=True)
    ]
    ax.set_yticks(range(len(labels)), yticklabels)
    ax.set_xlabel("recovered")
    ax.set_ylabel("intended")
    if title:
        ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            cell = "n/a" if row_n[i, 0] == 0 else f"{norm[i, j]:.2f}"
            ax.text(j, i, cell, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy(
    tier_acc: dict[str, float],
    path: str | Path,
    chance: float | None = None,
    *,
    accuracy_ci: dict[str, list[float | None]] | None = None,
) -> None:
    from matplotlib.ticker import PercentFormatter

    plt = _pyplot()

    tiers = list(tier_acc)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(tiers, [tier_acc[t] for t in tiers], color="#4c72b0")
    if accuracy_ci:
        for i, tier in enumerate(tiers):
            bounds = accuracy_ci.get(tier)
            if bounds and all(b is not None and np.isfinite(b) for b in bounds):
                ax.vlines(i, bounds[0], bounds[1], color="#1e293b", linewidth=1.5)
                ax.hlines(bounds, i - 0.08, i + 0.08, color="#1e293b", linewidth=1.5)
        ax.set_title("Recovery accuracy · bootstrap 95% CI")
    ax.set_ylabel("Recovery accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1)
    for i, t in enumerate(tiers):
        ax.text(i, max(0.025, tier_acc[t] / 2), f"{tier_acc[t]:.1%}", ha="center", color="white")
    if chance is not None:
        ax.axhline(chance, ls="--", lw=0.9, color="#888", label=f"Chance: {chance:.1%}")
        ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_va_scatter(records, tier: str, path: str | Path) -> None:
    plt = _pyplot()

    sub = [r for r in records if r["tier"] == tier]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(
        [r["intended_valence"] for r in sub],
        [r["recovered_valence"] for r in sub],
        alpha=0.6,
        label="valence",
        color="#c44e52",
    )
    ax.scatter(
        [r["intended_arousal"] for r in sub],
        [r["recovered_arousal"] for r in sub],
        alpha=0.6,
        label="arousal",
        color="#55a868",
    )
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.8)
    ax.set_xlabel("intended")
    ax.set_ylabel("recovered")
    ax.set_title(f"valence/arousal ({tier})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
