"""
visualization_scripts/common.py — Shared plotting utilities: the style, where the
results live, smoothing, and the recurring figure pieces (min-max bands, a legend
below the axes, saving).
"""
from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotstyle


# ── Paths ─────────────────────────────────────────────────────────────────────
# Finished runs are moved off $HOME to the vault (not visible from the GPU nodes);
# long runs that do not fit in $HOME write to $WORK. A results folder is looked up
# in that order.

RESULTS = Path(__file__).resolve().parent.parent / "results"
VAULT = Path(os.environ.get("HPCVAULT", "/home/vault/v170be/v170be10")) / "plasticity_results"
WORK = Path(os.environ.get("WORK", "/home/atuin/v170be/v170be10")) / "plasticity_results"


def results_dir(name: str, pattern: str = "*") -> Path:
    """
    The first of RESULTS/name, VAULT/name, WORK/name holding something that matches
    `pattern`; RESULTS/name when none does.
    """
    for root in (RESULTS, VAULT, WORK):
        if glob.glob(str(root / name / pattern)):
            return root / name
    return RESULTS / name


# The two task streams of experiments/run_tasks_stream.py, MLP 2x30, SGD.
PERM_STREAM = "shapes3d_stream_perm100_sgd"          # permuted pixels: input drift
FEATURE_STREAM = "shapes3d_stream_features_sgd"      # feature switches: output drift
WIDTH_STREAM = "shapes3d_stream_features_width_sgd"  # feature switches, one run per width and seed


def stream_dir(name: str) -> Path:
    """Results folder of a task stream: the one holding its run directories."""
    return results_dir(name, "stream_*_seed*")


def task_streams() -> list[tuple[Path, str]]:
    """(results folder, title) of the permuted-pixel and the feature-switch stream."""
    return [(stream_dir(PERM_STREAM), r"Permuted pixels (azimuth parity)"),
            (stream_dir(FEATURE_STREAM), r"Feature switches (parity)")]


# ── Style ─────────────────────────────────────────────────────────────────────
# Everything below is built on the PyPI `plotstyle` package: the Nature preset
# with the notebook overlay (fonts sized for screen-size figures), a grid, and
# the colourblind-safe Okabe-Ito palette.

STYLE = ["nature", "notebook", "grid", "okabe-ito"]

# (color, linestyle, marker) per series; linestyle and marker vary with colour
# so lines stay distinguishable without hue.
_SERIES = plotstyle.palette("nature", with_markers=True)
PALETTE = [c for c, _, _ in _SERIES]
LINESTYLES = [ls for _, ls, _ in _SERIES]
MARKERS = [m for _, _, m in _SERIES]
CMAP = plt.get_cmap("viridis")

# Font sizes of the single-chart figures made for slides.
TITLE_SIZE = 24
LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 18


def set_style() -> None:
    """Apply the plotting style globally. Call once before creating figures."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Helvetica/Arial missing on the cluster
        plotstyle.use(STYLE)
    # Scripts lay out with fig.tight_layout(rect=...), which constrained layout overrides.
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["axes.titlesize"] = plt.rcParams["axes.labelsize"]  # the overlay leaves Nature's 7pt titles


def style_for(i: int) -> dict:
    """color/linestyle kwargs for the i-th series."""
    color, linestyle, _ = _SERIES[i % len(_SERIES)]
    return {"color": color, "linestyle": linestyle}


def ordered_colors(n: int):
    """n colours for an ordered quantity (e.g. widths): dark -> light."""
    return CMAP(np.linspace(0.0, 0.85, n))


def bold_suptitle(fig, text: str, **kwargs):
    kwargs.setdefault("fontweight", "bold")
    kwargs.setdefault("fontsize", plt.rcParams["font.size"] + 4)
    return fig.suptitle(text, **kwargs)


def slide_ticks(ax) -> None:
    """Large ticks of the slide figures."""
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, length=7, width=1.4)
    ax.tick_params(axis="both", which="minor", length=4, width=1.0)


# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average, same length, zero-padded at the edges."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def moving_average(y, window: int) -> np.ndarray:
    """
    Moving average, same length, that shrinks the window at the edges instead of
    padding with zeros and skips NaNs (NaN only where the whole window is NaN).
    """
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y)
    num = np.convolve(np.where(ok, y, 0), np.ones(window), mode="same")
    den = np.convolve(ok.astype(float), np.ones(window), mode="same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def moving_average_valid(rows, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    (x, ma): moving average of every row of `rows` (seeds, tasks) over `window` tasks,
    'valid' mode (no edge padding), and the task index at the centre of each window.
    """
    kernel = np.ones(window) / window
    ma = np.stack([np.convolve(r, kernel, mode="valid") for r in rows])
    return np.arange(ma.shape[1]) + (window - 1) / 2, ma


def first_below(y, threshold) -> Optional[int]:
    """Index of the first value of y below threshold, or None."""
    i = np.flatnonzero(np.asarray(y) < threshold)
    return int(i[0]) if len(i) else None


# ── Figure pieces ─────────────────────────────────────────────────────────────

def band(ax, x, rows, color, label=None, alpha=0.18, **line_kw):
    """Mean over seeds of `rows` (seeds, x) as a line over a min-max band."""
    rows = np.asarray(rows)
    ax.fill_between(x, rows.min(0), rows.max(0), color=color, alpha=alpha, linewidth=0)
    ax.plot(x, rows.mean(0), color=color, label=label, **line_kw)


def legend_below(fig, handles, labels, bottom: float, ncol: Optional[int] = None, **kwargs) -> None:
    """One legend for the whole figure, centred below the axes, which keep the top 1 - bottom."""
    fig.legend(handles, labels, loc="lower center", ncol=ncol or len(labels), bbox_to_anchor=(0.5, 0.0),
               frameon=False, **kwargs)
    fig.tight_layout(rect=(0, bottom, 1, 1))


def save(fig, out) -> Path:
    """Write the figure as a 150-dpi PNG, close it and print where it went."""
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")
    return Path(out)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
