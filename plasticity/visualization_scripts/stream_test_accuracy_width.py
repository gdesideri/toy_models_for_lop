"""
visualization_scripts/stream_test_accuracy_width.py — Final test accuracy of every
task of the feature-switch stream, one line per MLP width (capacity). Lines are the
mean over seeds, bands the min-max over seeds. Reads each task's summary.json, so
runs that stopped early (e.g. on a quota error) are plotted up to the last finished task.

    default   two panels: (a) per-task values, (b) moving average over WINDOW tasks
    --ma      one slide-size chart: the moving average only, up to task --max-task

Run with:  uv run python -m visualization_scripts.stream_test_accuracy_width [stream results dir]
               [--window N] [--ma [--max-task N]]
           (default: results/shapes3d_stream_features_width_sgd, window 25, max task 500)
Writes   : results/stream_test_accuracy_width.png      (default)
           results/stream_test_accuracy_width_ma.png   (--ma)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization_scripts import common


def width_of(run) -> int:
    """Hidden width of an mlp_<layers>x<width> run, from its directory name."""
    return int(re.search(r"_mlp_\d+x(\d+)_", str(run)).group(1))


def load_run(run):
    """Final test accuracy of each finished task of one run, in task order."""
    acc = []
    for t in sorted(glob.glob(f"{run}/task_*")):
        try:
            with open(f"{t}/summary.json") as f:
                acc.append(json.load(f)["final_test_acc"])
        except (FileNotFoundError, json.JSONDecodeError):
            break                                                 # task never finished
    return np.array(acc)


def load_by_width(stream_dir):
    """{width: (seeds, tasks) final test accuracy}, truncated to the shortest seed."""
    runs = defaultdict(list)
    for run in sorted(glob.glob(str(Path(stream_dir) / "stream_*_seed*"))):
        runs[width_of(run)].append(load_run(run))
    out = {}
    for width in sorted(runs):
        n_tasks = min(len(a) for a in runs[width])
        out[width] = np.stack([a[:n_tasks] for a in runs[width]])
    return out


def plot_panels(acc_by_width, window):
    fig, (ax_raw, ax_ma) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for width, color in zip(acc_by_width, common.ordered_colors(len(acc_by_width))):
        acc = acc_by_width[width]
        x = np.arange(acc.shape[1])
        label = rf"$2 \times {width}$ ($n = {len(acc)}$ seeds, {len(x)} tasks)"
        ax_raw.plot(x, acc.mean(0), color=color, linewidth=0.8, label=label)
        common.band(ax_ma, *common.moving_average_valid(acc, window), color=color, linewidth=1.5)
        print(f"width {width}: {len(acc)} seeds, {len(x)} tasks, "
              f"mean test acc first/last 50 tasks = {acc[:, :50].mean():.3f} / {acc[:, -50:].mean():.3f}")
    for ax in (ax_raw, ax_ma):
        ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1.2, label=r"chance ($0.5$)" if ax is ax_raw else None)
        ax.set_ylabel(r"final test accuracy")
    ax_raw.set_title(r"(a) per task")
    ax_ma.set_title(rf"(b) moving average over {window} tasks")
    ax_ma.set_xlabel(r"task index")
    common.bold_suptitle(fig, r"Final test accuracy of each task by MLP width, feature-switch stream, SGD (mean and range over seeds)")
    common.legend_below(fig, *ax_raw.get_legend_handles_labels(), bottom=0.05)
    common.save(fig, common.RESULTS / "stream_test_accuracy_width.png")


def plot_ma(acc_by_width, window, max_task):
    fig, ax = plt.subplots(figsize=(16, 8))
    for width, color in zip(acc_by_width, common.ordered_colors(len(acc_by_width))):
        acc = acc_by_width[width]
        x, ma = common.moving_average_valid(acc, window)
        keep = x <= max_task
        common.band(ax, x[keep], ma[:, keep], color=color, linewidth=2.2, label=rf"$2 \times {width}$")
        print(f"width {width}: {len(acc)} seeds, {acc.shape[1]} tasks, shown up to task {x[keep][-1]:g}")
    ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1.6, label=r"chance ($0.5$)")
    ax.set_xlim(0, max_task)
    ax.set_xlabel(r"task index", fontsize=common.LABEL_SIZE)
    ax.set_ylabel(r"final test accuracy", fontsize=common.LABEL_SIZE)
    common.slide_ticks(ax)
    common.bold_suptitle(fig, "Final test accuracy of each task, MLP width, feature-switch stream, SGD\n"
                              rf"moving average over {window} tasks", fontsize=common.TITLE_SIZE)
    common.legend_below(fig, *ax.get_legend_handles_labels(), bottom=0.07,
                        fontsize=common.LEGEND_SIZE, columnspacing=1.4, handlelength=1.6)
    common.save(fig, common.RESULTS / "stream_test_accuracy_width_ma.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stream_dir", nargs="?", type=Path, default=None)
    p.add_argument("--window", type=int, default=25, help="moving-average window (tasks)")
    p.add_argument("--ma", action="store_true", help="slide chart of the moving average only")
    p.add_argument("--max-task", type=int, default=500, help="last task index shown (--ma)")
    args = p.parse_args()

    acc_by_width = load_by_width((args.stream_dir or common.stream_dir(common.WIDTH_STREAM)).resolve())
    common.set_style()
    if args.ma:
        plot_ma(acc_by_width, args.window, args.max_task)
    else:
        plot_panels(acc_by_width, args.window)


if __name__ == "__main__":
    main()
