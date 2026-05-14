"""
visualization_scripts/common.py — Shared plotting utilities.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

def set_style() -> None:
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       11,
        "axes.titlesize":  12,
        "axes.labelsize":  11,
        "legend.fontsize": 9,
        "figure.dpi":      120,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def steps_and_values(log: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    steps  = np.array([e["step"]  for e in log if key in e])
    values = np.array([e[key]     for e in log if key in e])
    return steps, values


def add_task_switch_vlines(ax: plt.Axes, switch_steps: Sequence[int], **kwargs) -> None:
    defaults = dict(color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    defaults.update(kwargs)
    for s in switch_steps:
        ax.axvline(s, **defaults)


# ── Learning-curve panel ──────────────────────────────────────────────────────

def plot_learning_curves(
    logs:          dict[str, list[dict]],
    metric:        str = "current_acc",
    title:         str = "",
    xlabel:        str = "Training step",
    ylabel:        str = "Accuracy",
    smooth_window: int = 1,
    switch_steps:  Optional[Sequence[int]] = None,
    ax:            Optional[plt.Axes] = None,
    save_path:     Optional[str] = None,
) -> plt.Figure:
    """
    Overlay learning curves for multiple runs / conditions.

    Args:
        logs:  Dict[run_label → list of metric dicts]
        metric: key to plot from each metric dict
    """
    set_style()
    fig, ax_ = (plt.figure(figsize=(7, 4)), plt.gca()) if ax is None else (ax.figure, ax)

    for i, (label, log) in enumerate(logs.items()):
        steps, vals = steps_and_values(log, metric)
        if len(steps) == 0:
            continue
        vals_s = smooth(vals, smooth_window)
        ax_.plot(steps, vals_s, color=PALETTE[i % len(PALETTE)], label=label)

    if switch_steps:
        add_task_switch_vlines(ax_, switch_steps)

    ax_.set_xlabel(xlabel)
    ax_.set_ylabel(ylabel)
    ax_.set_title(title)
    if len(logs) > 1:
        ax_.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Heatmap ───────────────────────────────────────────────────────────────────

def plot_heatmap(
    matrix:      np.ndarray,
    row_labels:  Sequence[str],
    col_labels:  Sequence[str],
    title:       str = "",
    fmt:         str = ".2f",
    cmap:        str = "viridis",
    save_path:   Optional[str] = None,
) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels)), max(3, len(row_labels))))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:{fmt}}",
                    ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] < 0.5 else "black")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Bar chart ─────────────────────────────────────────────────────────────────

def plot_bar(
    values:    dict[str, float],
    title:     str = "",
    xlabel:    str = "",
    ylabel:    str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(max(4, len(values)), 4))
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    bars = ax.bar(keys, vals, color=PALETTE[:len(keys)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
