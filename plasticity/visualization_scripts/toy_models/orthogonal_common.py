"""Shared style of the orthogonal-tasks figures (animate_orthogonal_tasks, plot_orthogonal_summary)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

UNIT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
TASK_TINTS = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00"]
SEED_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#D55E00"]
LABEL0, LABEL1 = "#9ecae1", "#E69F00"          # points with label 0 / 1
CONTINUED, FRESH = "#000000", "#D55E00"


def unit_colors(h: int) -> list:
    if h <= len(UNIT_COLORS):
        return UNIT_COLORS[:h]
    return [plt.get_cmap("tab20")(i % 20) for i in range(h)]


def log_index(steps, step) -> int:
    """First logged state at or after `step`."""
    return int(min(np.searchsorted(steps, step), len(steps) - 1))


def shade_tasks(ax, schedule, S) -> None:
    """Tint each phase by its task and mark the switches."""
    for q, task in enumerate(schedule):
        ax.axvspan(q * S, (q + 1) * S, color=TASK_TINTS[task % len(TASK_TINTS)], alpha=0.08, lw=0)
        if q:
            ax.axvline(q * S, color="0.6", lw=0.5)
