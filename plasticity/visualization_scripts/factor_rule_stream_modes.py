"""
visualization_scripts/factor_rule_stream_modes.py — Current-rule test accuracy through the
800-task factor-rule stream for the three adaptation modes of the pretrained mlp_3x<n>:
frozen encoder (head only), last hidden layer + head, full network + head. Only the rule
being trained is scored (not the mean over all six rules). Lines are the mean over seeds,
bands the min-max over seeds.

    default    (a) mean over each 6-task cycle (one task per factor), so the factor-to-factor
                   difficulty swing within a cycle does not hide the trend
               (b) one panel per factor: the accuracy on every task of that factor
    --single   one slide-size chart, one line per mode; each point is a moving average over
               WINDOW consecutive tasks (default 6 = one cycle; --window 1 for the raw values)

Run with:  uv run python -m visualization_scripts.factor_rule_stream_modes [--steps 500|10ep] [--out PNG]
           uv run python -m visualization_scripts.factor_rule_stream_modes --single [--steps 500|10ep] [--window N] [--out PNG]
           (default steps: 500, with --single 10ep)
Reads    : <mode dir>/factorrule_shapes3d_mlp_3x30_perm000_seed*/task_end_metrics.csv
Writes   : results/factor_rule_stream_modes_<steps>.png          (default)
           results/factor_rule_stream_modes_single_<steps>.png   (--single)
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization_scripts import common
from tasks.factor_rules import FACTORS, N_FACTORS

# steps per task -> (long title, short title, results dir suffix)
STEPS = {"500": ("500 SGD steps per task", "500 steps per task", ""),
         "10ep": ("15188 SGD steps (about 10 epochs) per task", "10 epochs per task", "_10ep_per_task")}
METRICS = "factorrule_*_seed*/task_end_metrics.csv"
# linearpurple / nonlineargreen / fullred of the write-up
MODE_COLORS = {"frozen encoder (head only)": "#B16AF7",
               "last hidden layer + head": "#3FAE6B",
               "full network + head": "#F76262"}


def mode_dirs(steps):
    """{mode: results dir} of each adaptation mode, wherever its finished runs are."""
    base = f"shapes3d_factor_rule_stream{STEPS[steps][2]}"
    return {"frozen encoder (head only)": common.results_dir(base, METRICS),
            "last hidden layer + head": common.results_dir(f"{base}_unfreeze_last", METRICS),
            "full network + head": common.results_dir(f"{base}_unfreeze_all", METRICS)}


def load_task_end_metrics(mode_dir):
    """Per seed, the rows of task_end_metrics.csv, cut to the shortest seed; None without finished runs."""
    files = sorted(glob.glob(str(Path(mode_dir) / METRICS)))
    if not files:
        return None
    rows = []
    for p in files:
        with open(p) as f:
            rows.append(list(csv.DictReader(f)))
    n = min(len(r) for r in rows)
    return [r[:n] for r in rows]


def load_mode(mode_dir):
    """(seeds, tasks) current-rule test accuracy of the finished runs, or None."""
    rows = load_task_end_metrics(mode_dir)
    if rows is None:
        return None
    return np.array([[float(x["current_task_test_accuracy"]) for x in r] for r in rows])


def plot_panels(data, steps):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1], hspace=0.5, wspace=0.08)
    ax_cyc = fig.add_subplot(gs[0, :])
    ax_f = [fig.add_subplot(gs[1 + f // 3, f % 3]) for f in range(N_FACTORS)]

    for i, (mode, acc) in enumerate(data.items()):
        if acc is None:
            continue
        style = dict(**common.style_for(i), linewidth=1.6)
        n_cyc = acc.shape[1] // N_FACTORS
        cyc = acc[:, :n_cyc * N_FACTORS].reshape(len(acc), n_cyc, N_FACTORS).mean(2)
        common.band(ax_cyc, np.arange(n_cyc) * N_FACTORS, cyc, label=rf"{mode} ($n = {len(acc)}$ seeds)", **style)
        for f, ax in enumerate(ax_f):
            common.band(ax, np.arange(f, acc.shape[1], N_FACTORS), acc[:, f::N_FACTORS], **style)
        print(f"{mode}: {len(acc)} seeds x {acc.shape[1]} tasks, mean acc first / last cycle "
              f"{cyc[:, 0].mean():.3f} / {cyc[:, -1].mean():.3f}")

    for ax in [ax_cyc] + ax_f:
        ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1.2)
        ax.set_ylim(0.35, 1.03)
    ax_cyc.text(798, 0.505, "chance", ha="right", va="bottom", color="0.35", fontsize=10)
    ax_cyc.set_title(r"(a) mean over each cycle of six tasks (one per factor)")
    ax_cyc.set_xlabel(r"task index (start of cycle)")
    ax_cyc.set_ylabel(r"current-rule test accuracy")
    ax_cyc.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=False)
    for f, ax in enumerate(ax_f):
        ax.set_title(rf"(b) {FACTORS[f].replace('_', ' ')}", fontsize=12)
        if f % 3:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"test accuracy")
        if f >= 3:
            ax.set_xlabel(r"task index")
    common.bold_suptitle(fig, rf"Factor-rule stream, pretrained mlp_3x30, SGD lr 0.01, {STEPS[steps][0]}"
                              "\n(mean and min-max over seeds)")
    return fig


def plot_single(data, steps, window):
    fig, ax = plt.subplots(figsize=(16, 8))
    for mode, acc in data.items():
        if acc is None:
            continue
        x, ma = common.moving_average_valid(acc, window)      # per seed, then mean / range
        common.band(ax, x, ma, color=MODE_COLORS[mode], alpha=0.2, linewidth=2.6, label=mode)
        print(f"{mode}: {len(acc)} seeds x {acc.shape[1]} tasks, mean acc first / last 6 tasks "
              f"{acc[:, :6].mean():.3f} / {acc[:, -6:].mean():.3f}; per seed, last 6 tasks "
              f"{np.round(acc[:, -6:].mean(1), 3).tolist()}")
    ax.axhline(0.5, color="0.4", linestyle=":", linewidth=1.6, label=r"chance ($0.5$)")
    ax.set_xlim(0, 800)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("task index", fontsize=common.LABEL_SIZE)
    ax.set_ylabel("final test accuracy", fontsize=common.LABEL_SIZE)
    common.slide_ticks(ax)
    avg = "" if window == 1 else f"\n{window}-task moving average, mean and range over seeds"
    common.bold_suptitle(fig, f"{STEPS[steps][1]}: test accuracy on the current task{avg}",
                         fontsize=common.TITLE_SIZE)
    common.legend_below(fig, *ax.get_legend_handles_labels(), bottom=0.07,
                        fontsize=common.LEGEND_SIZE, columnspacing=1.4, handlelength=1.8)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true", help="one slide chart with a moving average")
    p.add_argument("--steps", choices=list(STEPS), default=None, help="default: 500, with --single 10ep")
    p.add_argument("--window", type=int, default=6, help="moving-average window in tasks (--single)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    steps = args.steps or ("10ep" if args.single else "500")

    data = {m: load_mode(d) for m, d in mode_dirs(steps).items()}
    for mode, acc in data.items():
        if acc is None:
            print(f"{mode}: no finished runs")
    common.set_style()
    if args.single:
        fig = plot_single(data, steps, args.window)
        name = f"factor_rule_stream_modes_single_{steps}.png"
    else:
        fig = plot_panels(data, steps)
        name = f"factor_rule_stream_modes_{steps}.png"
    common.save(fig, args.out or common.RESULTS / name)


if __name__ == "__main__":
    main()
