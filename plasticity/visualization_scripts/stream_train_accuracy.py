"""
visualization_scripts/stream_train_accuracy.py — Final train accuracy of every
task of the task streams (the last epoch of each task), both families in one line
chart. Lines are the mean over seeds, bands the min-max over seeds. Reads each
task's metrics.csv.

Run with:  uv run python -m visualization_scripts.stream_train_accuracy [stream results dirs ...]
           (default: the permutation and the feature-switch streams, SGD)
Writes   : results/stream_train_accuracy.png
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization_scripts import common


def load_final_train_acc(stream_dir):
    """(seeds, tasks) train accuracy at the last epoch of each task, and the task indices."""
    curves = []
    for run in sorted(glob.glob(str(stream_dir / "stream_*_seed*"))):
        acc = []
        for t in sorted(glob.glob(f"{run}/task_*")):
            with open(f"{t}/metrics.csv") as f:
                acc.append([float(r["train_acc"]) for r in csv.DictReader(f)])
        curves.append(np.array(acc))                              # (tasks, epochs)
    n_tasks = min(c.shape[0] for c in curves)
    return np.arange(n_tasks), np.stack([c[:n_tasks, -1] for c in curves])


def main():
    dirs = [(Path(d).resolve(), Path(d).name) for d in sys.argv[1:]] or common.task_streams()
    common.set_style()
    fig, ax = plt.subplots(figsize=(16, 5.5))
    for i, (stream_dir, title) in enumerate(dirs):
        x, acc = load_final_train_acc(stream_dir)
        common.band(ax, x, acc, **common.style_for(i), linewidth=1.1, label=title + rf" ($n = {len(acc)}$ seeds)")
        print(f"{stream_dir.name}: {len(acc)} seeds, {len(x)} tasks")
    ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1.2, label=r"chance ($0.5$)")
    ax.set_xlabel(r"task index")
    ax.set_ylabel(r"final train accuracy")
    ax.set_ylim(0.45, 1.02)
    common.bold_suptitle(fig, r"Final train accuracy of each task, MLP $2 \times 30$, SGD (mean and range over seeds)")
    common.legend_below(fig, *ax.get_legend_handles_labels(), bottom=0.08)
    common.save(fig, common.RESULTS / "stream_train_accuracy.png")


if __name__ == "__main__":
    main()
