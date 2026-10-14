"""
visualization_scripts/factor_rule_stream_alive_units.py — Hidden units still alive through
the 800-task factor-rule stream, for the two adaptation modes that train hidden layers:

    left   full network + head: one line per hidden layer (1, 2, 3)
    right  last hidden layer + head: the trained layer (3) only

A unit is alive at the end of a task if its ReLU output is > 0 on at least one test image
(width minus layer<L>_dead_units_test of task_end_metrics.csv). Each point is a moving
average over WINDOW consecutive tasks (default 6 = one cycle of the six factors; --window 1
for the raw per-task values). Lines are the mean over seeds, bands the min-max over seeds.

Run with:  uv run python -m visualization_scripts.factor_rule_stream_alive_units [--steps 500|10ep] [--window N] [--out PNG]
Reads    : <mode dir>/factorrule_shapes3d_mlp_3x30_perm000_seed*/task_end_metrics.csv
Writes   : results/factor_rule_stream_alive_units_<steps>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization_scripts import common
from visualization_scripts.factor_rule_stream_modes import STEPS, load_task_end_metrics, mode_dirs

WIDTH = 30                          # units per hidden layer of mlp_3x30
PANELS = [("full network + head", (1, 2, 3)),
          ("last hidden layer + head", (3,))]


def load_alive(mode_dir, layers):
    """{layer: (seeds, tasks) alive units} of the finished runs, or None."""
    rows = load_task_end_metrics(mode_dir)
    if rows is None:
        return None
    return {L: np.array([[WIDTH - float(x[f"layer{L}_dead_units_test"]) for x in r] for r in rows])
            for L in layers}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", choices=list(STEPS), default="500")
    p.add_argument("--window", type=int, default=6, help="moving-average window (tasks)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    dirs = mode_dirs(args.steps)
    colors = dict(zip((1, 2, 3), common.CMAP([0.0, 0.42, 0.8])))    # layer 3 has the same colour in both panels

    common.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)
    for ax, (mode, layers) in zip(axes, PANELS):
        alive = load_alive(dirs[mode], layers)
        if alive is None:
            print(f"{mode}: no finished runs")
            continue
        for L in layers:
            common.band(ax, *common.moving_average_valid(alive[L], args.window), color=colors[L], alpha=0.2,
                        linewidth=2.6, label=f"layer {L}")
            print(f"{mode} layer {L}: alive per seed after task 0 {alive[L][:, 0].astype(int).tolist()}, "
                  f"task 799 {alive[L][:, -1].astype(int).tolist()}")
        ax.set_title(mode, fontsize=common.LABEL_SIZE, pad=12)
        ax.set_xlim(0, 800)
        ax.set_ylim(0, WIDTH + 1)
        ax.set_yticks(np.arange(0, WIDTH + 1, 5))
        ax.set_xlabel("task index", fontsize=common.LABEL_SIZE)
        common.slide_ticks(ax)
    axes[0].set_ylabel(f"units alive (of {WIDTH})", fontsize=common.LABEL_SIZE)
    avg = "" if args.window == 1 else f", {args.window}-task moving average"
    common.bold_suptitle(fig, f"{STEPS[args.steps][1]}: alive units per layer{avg}", fontsize=common.TITLE_SIZE)
    common.legend_below(fig, *axes[0].get_legend_handles_labels(), bottom=0.07,
                        fontsize=common.LEGEND_SIZE, columnspacing=1.6, handlelength=1.8)
    common.save(fig, args.out or common.RESULTS / f"factor_rule_stream_alive_units_{args.steps}.png")


if __name__ == "__main__":
    main()
