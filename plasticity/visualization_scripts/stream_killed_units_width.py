"""
visualization_scripts/stream_killed_units_width.py — Hidden units killed by the
optimizer during each task of the feature-switch width sweep, and the units
still alive, for one hidden layer (layer 1 by default), one line per MLP width.

Killed during task t (= O_t of stream_unit_turnover.py): active at the start of
task t (the end-of-task-(t-1) weights; the initial network for task 0), dead after
training on t. Active = ReLU output > 0 on at least one probe image; the probe is
N_PROBE images of the run's own training split, drawn as in stream_unit_turnover.py.
In the feature-switch stream consecutive tasks share their images, so this is the
only way a unit dies.

    default   (a) units killed per task (thin: mean over seeds; thick: that mean smoothed
              over WINDOW tasks), (b) units alive at the end of each task (mean over seeds,
              band = min-max over seeds)
    --ma      one slide-size chart: units alive, each seed averaged over WINDOW tasks,
              up to task --max-task

Both mark where the test accuracy starts to drop: the first task at which the WINDOW-task
average of the final test accuracy (mean over seeds, read by stream_test_accuracy_width.py)
falls below ELBOW_ACC, labelled with the units alive there. Runs that stopped early are
plotted up to their last finished task.

Measuring runs on a GPU, one run per call (experiments/launch_scripts/submit_killed_units_width.sh):
    uv run python -m visualization_scripts.stream_killed_units_width --compute <run dir> [--n_probe N]
Plotting reads the per-run files:
    uv run python -m visualization_scripts.stream_killed_units_width [layer] [--n_probe N] [--window N]
        [--ma [--max-task N]]
Writes   : results/shapes3d_stream_features_width_sgd_killed_units/killed_n<N>_<run>.json
           results/stream_killed_units_width_n<N>_layer<l>.png   (default)
           results/stream_alive_units_width_ma_n<N>_layer<l>.png (--ma)
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from visualization_scripts import common
from visualization_scripts.stream_test_accuracy_width import load_by_width as load_test_acc, width_of

KILLED_DIR = common.RESULTS / f"{common.WIDTH_STREAM}_killed_units"   # the stream dir's filesystem is full
N_PROBE = 20_000
PROBE_SALT = 29                     # the probe of stream_unit_turnover.py
WINDOW = 25
ELBOW_ACC = 0.99                    # accuracy drop onset: smoothed test accuracy below this


# ── measuring ─────────────────────────────────────────────────────────────────

def compute_run(run, device, n_probe=N_PROBE):
    from utils.checkpoints import load_checkpoint
    from utils.evaluation import alive_masks_over
    from utils.stream_probe import (draw_probe, load_init_model, task_batches, task_checkpoints, task_dirs,
                                    task_perm, task_training_images)

    run = Path(run)
    seed = json.load(open(run / "args.json"))["seed"]
    t0 = time.time()
    _, images = task_training_images(run, device)
    _, probe = draw_probe(images, seed, n_probe, PROBE_SALT)
    n_probe = len(probe)
    del images

    model = load_init_model(run, device)
    out = {"run": run.name, "width": width_of(run), "seed": seed, "n_probe": n_probe,
           "measured_on": f"{n_probe} images of the run's training split, task permutation applied",
           "task": [], "killed": {}, "alive_start": {}, "alive_end": {}}
    for n, t in enumerate(task_dirs(run)):
        ckpts = task_checkpoints(t)
        if not (t / "summary.json").exists() or not ckpts:
            break                                                 # task never finished
        perm = task_perm(t)
        start = alive_masks_over(model, task_batches(probe, perm))   # weights at the end of task t-1
        load_checkpoint(ckpts[-1], model)
        end = alive_masks_over(model, task_batches(probe, perm))
        out["task"].append(n)
        for k in range(len(start)):
            key = f"layer_{k + 1}"
            out["killed"].setdefault(key, []).append(int((start[k] & ~end[k]).sum()))
            out["alive_start"].setdefault(key, []).append(int(start[k].sum()))
            out["alive_end"].setdefault(key, []).append(int(end[k].sum()))
    out_dir = run.parent.parent / f"{run.parent.name}_killed_units"   # next to the stream dir, not inside it
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / f"killed_n{n_probe}_{run.name}.json", "w"))
    print(f"{run.name}: {len(out['task'])} tasks in {time.time() - t0:.0f}s")


# ── plotting ──────────────────────────────────────────────────────────────────

def load_killed(n_probe):
    """{width: [per-run measurement]} from the files written by --compute."""
    files = sorted(glob.glob(str(KILLED_DIR / f"killed_n{n_probe}_*.json")))
    if not files:
        raise SystemExit(f"no killed_n{n_probe}_*.json in {KILLED_DIR}: "
                         f"run N_PROBE={n_probe} sbatch experiments/launch_scripts/submit_killed_units_width.sh")
    by_width = defaultdict(list)
    for f in files:
        r = json.load(open(f))
        by_width[r["width"]].append(r)
    return dict(sorted(by_width.items()))


def accuracy_drop(test_acc, window, n):
    """First task (< n) where the window-task average of the mean test accuracy is below ELBOW_ACC."""
    t = common.first_below(common.moving_average(test_acc.mean(0), window), ELBOW_ACC)
    return t if t is not None and t < n else None


def mark_drop(ax, t, a, color, markersize, edge, offset, fontsize, stroke):
    """A dot on the alive-units line at the accuracy drop, labelled with the units alive."""
    ax.plot(t, a, marker="o", markersize=markersize, color=color, markeredgecolor="white",
            markeredgewidth=edge, zorder=5)
    ax.annotate(f"{a:.0f}", (t, a), xytext=(-offset, 0), textcoords="offset points", ha="right",
                va="center", fontsize=fontsize, color="0.15", zorder=6,
                path_effects=[pe.withStroke(linewidth=stroke, foreground="white")])


def plot_panels(by_width, test_acc, layer, n_probe, window):
    key = f"layer_{layer}"
    fig, (ax_killed, ax_alive) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for (width, runs), color in zip(by_width.items(), common.ordered_colors(len(by_width))):
        n = min(len(r["task"]) for r in runs)
        tasks = np.arange(n)
        killed = np.mean([r["killed"][key][:n] for r in runs], axis=0)
        alive = np.array([r["alive_end"][key][:n] for r in runs])
        t = accuracy_drop(test_acc[width], window, n)
        label = rf"$2 \times {width}$ ($n = {len(runs)}$ seeds, {n} tasks)"
        if t is not None:
            a = alive.mean(0)[t]
            label += rf"; drop at $t = {t}$, ${a:.0f}$ alive"
            print(f"width {width}: test accuracy drops below {ELBOW_ACC} at task {t}, {a:.0f} units alive")
        ax_killed.plot(tasks, killed, color=color, linewidth=0.6, alpha=0.3)
        ax_killed.plot(tasks, common.moving_average(killed, window), color=color, linewidth=2.0, label=label)
        common.band(ax_alive, tasks, alive, color=color, linewidth=2.0)
        if t is not None:
            mark_drop(ax_alive, t, a, color, markersize=9, edge=2, offset=8,
                      fontsize=plt.rcParams["font.size"] - 1, stroke=3)
        print(f"width {width}: {len(runs)} seeds, {n} tasks, {key} killed in total (mean over seeds) "
              f"{np.mean([sum(r['killed'][key][:n]) for r in runs]):.0f}, alive at the end {alive[:, -1].tolist()}")
    ax_killed.set_title(r"(a) units killed during task $t$" + rf" (thick: ${window}$-task average)")
    ax_killed.set_ylabel(r"units killed")
    ax_alive.set_title(r"(b) units alive at the end of task $t$ (band: min-max over seeds; "
                       rf"$\bullet$: test accuracy starts to drop, ${window}$-task average $< {ELBOW_ACC}$, labelled with the units alive)")
    ax_alive.set_ylabel(r"units alive")
    ax_alive.set_xlabel(r"task index $t$")
    common.bold_suptitle(fig, rf"Layer-${layer}$ units killed by the optimizer and still alive, feature-switch stream, SGD "
                              rf"(mean over seeds; probe: ${n_probe // 1000}$k training images)")
    common.legend_below(fig, *ax_killed.get_legend_handles_labels(), bottom=0.09, ncol=2)
    common.save(fig, common.RESULTS / f"stream_killed_units_width_n{n_probe}_layer{layer}.png")


def plot_ma(by_width, test_acc, layer, n_probe, window, max_task):
    key = f"layer_{layer}"
    fig, ax = plt.subplots(figsize=(16, 8))
    for (width, runs), color in zip(by_width.items(), common.ordered_colors(len(by_width))):
        n = min(len(r["task"]) for r in runs)
        alive = np.stack([common.moving_average(r["alive_end"][key][:n], window) for r in runs])
        t = accuracy_drop(test_acc[width], window, n)
        shown = min(n, max_task + 1)
        common.band(ax, np.arange(shown), alive[:, :shown], color=color, linewidth=2.4, label=rf"$2 \times {width}$")
        if t is not None and t <= max_task:
            a = alive.mean(0)[t]
            mark_drop(ax, t, a, color, markersize=13, edge=2.5, offset=11, fontsize=common.TICK_SIZE - 2, stroke=4)
            print(f"width {width}: test accuracy drops below {ELBOW_ACC} at task {t}, {a:.0f} units alive")
    ax.set_xlim(0, max_task)
    ax.set_xlabel(r"task index", fontsize=common.LABEL_SIZE)
    ax.set_ylabel(r"units alive", fontsize=common.LABEL_SIZE)
    common.slide_ticks(ax)
    common.bold_suptitle(fig, f"Layer-{layer} alive units over time, {window} tasks average", fontsize=common.TITLE_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=11, color="0.35",
                          markeredgecolor="white", markeredgewidth=2))
    labels.append("test accuracy starts to drop")
    common.legend_below(fig, handles, labels, bottom=0.07, fontsize=common.LEGEND_SIZE,
                        columnspacing=1.2, handlelength=1.4, handletextpad=0.5)
    common.save(fig, common.RESULTS / f"stream_alive_units_width_ma_n{n_probe}_layer{layer}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("layer", nargs="?", type=int, default=1, help="hidden layer to plot")
    p.add_argument("--compute", type=str, default=None, help="run dir to measure")
    p.add_argument("--n_probe", type=int, default=N_PROBE, help="probe images per run")
    p.add_argument("--window", type=int, default=WINDOW, help="moving-average window (tasks)")
    p.add_argument("--ma", action="store_true", help="slide chart of the alive units only")
    p.add_argument("--max-task", type=int, default=500, help="last task index shown (--ma)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.compute:
        compute_run(a.compute, a.device, a.n_probe)
    else:
        by_width = load_killed(a.n_probe)
        test_acc = load_test_acc(common.stream_dir(common.WIDTH_STREAM))
        common.set_style()
        if a.ma:
            plot_ma(by_width, test_acc, a.layer, a.n_probe, a.window, a.max_task)
        else:
            plot_panels(by_width, test_acc, a.layer, a.n_probe, a.window)
