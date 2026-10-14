"""
visualization_scripts/drift_story.py — The four charts of the input-drift vs output-drift
argument, for the permuted-pixel stream (input drift: the images change, the label rule
does not) and the feature-switch stream (output drift: the images stay, the label changes).

    1  accuracy    final test accuracy Acc(t)
    2  alive       alive units A_1(t), A_2(t); T_2 = layer-2 death time
    3  turnover    R (revived at the switch), K (killed at the switch), O (killed by optimization)
    4  gradient    input drift only: gradient reaching layers 1 and 2 at the start of each
                   task; layer-1 units alive vs reached by the gradient vs partially active

Every line is the 25-task moving average of the mean over seeds; bands span the seeds.
T_c: first task where the moving average of final test accuracy drops below ACC_T.
T_2: first task where the moving average of alive layer-2 units drops below A2_T.

Reads (all from the stream result folders):
    summary.json, task_*/metrics.csv                   accuracy
    alive_units_seed<k>.json                           alive units, full training data (stream_alive_units.py)
    unit_turnover_n100000_seed<k>.json                 R / K / O (stream_unit_turnover.py)
    gradient_reach_n100000_seed<k>.json                gradients, layer-1 activity (stream_gradient_reach.py)
Run with:  uv run python -m visualization_scripts.drift_story
Writes   : results/drift_story_{1_accuracy,2_alive,3_turnover,4_gradient}.png
"""
from __future__ import annotations

import csv
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization_scripts import common

STREAMS = {"input drift (permuted pixels)": common.PERM_STREAM,
           "output drift (feature switches)": common.FEATURE_STREAM}
W = 25          # moving-average window, in tasks
ACC_T = 0.9     # collapse threshold on final test accuracy
A2_T = 10       # layer-2 death threshold on alive units (of 30)
REACH = 0.01    # a layer-1 unit is reached if its gradient is > REACH x the task-0 median


def smooth(y):
    return common.moving_average(y, W)


def first_below(y, thr):
    return common.first_below(smooth(y), thr)


def runs(stream_dir, pattern):
    files = sorted(glob.glob(str(stream_dir / pattern)))
    if not files:
        raise SystemExit(f"missing {stream_dir / pattern}")
    return files


def load(stream_dir):
    """Per seed arrays over the tasks; all cut to the shortest seed."""
    d = {}
    seeds = sorted(glob.glob(str(stream_dir / "stream_*_seed*")))
    acc = [[t["final_test_acc"] for t in json.load(open(f"{s}/summary.json"))["tasks"]] for s in seeds]
    ep1 = []
    for s in seeds:
        row = []
        for t in sorted(glob.glob(f"{s}/task_*")):
            with open(f"{t}/metrics.csv") as f:
                row.append(float(next(csv.DictReader(f))["test_acc"]))
        ep1.append(row)
    alive = [json.load(open(f)) for f in runs(stream_dir, "alive_units_seed*.json")]
    turn = [json.load(open(f)) for f in runs(stream_dir, "unit_turnover_n100000_seed*.json")]
    grad = [json.load(open(f)) for f in runs(stream_dir, "gradient_reach_n100000_seed*.json")]
    n = min(map(len, acc + ep1 + [a["task"] for a in alive] + [t["task"] for t in turn] + [g["task"] for g in grad]))
    cut = lambda rows: np.array([r[:n] for r in rows], dtype=float)
    nan = lambda v: np.nan if v is None else v
    d["acc"], d["ep1"] = cut(acc), cut(ep1)
    d["A1"], d["A2"] = cut([a["layer_1"] for a in alive]), cut([a["layer_2"] for a in alive])
    for q in "RKO":
        d[q] = cut([[nan(v) for v in t[q]["layer_1"]] for t in turn])
    d["g1"], d["g2"] = cut([g["g_1"] for g in grad]), cut([g["g_2"] for g in grad])
    ug = [np.array(g["unit_grad_1"])[:n] for g in grad]                     # (tasks, 30) per seed
    d["reach"] = cut([(u > REACH * np.median(u[0])).sum(1) for u in ug])
    rates = [np.array(g["rates_1"])[:n] for g in grad]
    d["alive_probe"] = cut([(r > 0).sum(1) for r in rates])
    d["partial"] = cut([((r >= 0.05) & (r <= 0.95)).sum(1) for r in rates])
    d["n"], d["seeds"] = n, len(seeds)
    d["T_c"], d["T_2"] = first_below(d["acc"].mean(0), ACC_T), first_below(d["A2"].mean(0), A2_T)
    return d


def band_line(ax, x, rows, style, label, log=False):
    sm = np.array([smooth(r) for r in rows])
    lo, hi = np.nanmin(sm, 0), np.nanmax(sm, 0)
    ax.fill_between(x, lo, hi, color=style["color"], alpha=0.15, linewidth=0)
    ax.plot(x, smooth(np.nanmean(rows, 0)), **style, linewidth=2.0, label=label)
    if log:
        ax.set_yscale("log")


def marker(ax, t, color, label):
    if t is not None:
        ax.axvline(t, color=color, linestyle=(0, (1, 1.5)), linewidth=1.6, label=label)


def finish(fig, axes, title, out, ncol=3):
    if title:
        common.bold_suptitle(fig, title)
    handles, labels, seen = [], [], set()
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); labels.append(l); seen.add(l)
    common.legend_below(fig, handles, labels, bottom=0.13 if len(labels) > ncol else 0.08, ncol=ncol)
    common.save(fig, common.RESULTS / out)


def main():
    data = {name: load(common.stream_dir(d)) for name, d in STREAMS.items()}
    colors = {name: common.style_for(i) for i, name in enumerate(data)}
    for name, d in data.items():
        print(f"{name}: {d['seeds']} seeds, {d['n']} tasks | T_c = {d['T_c']} | T_2 = {d['T_2']}")
    common.set_style()

    # 1 accuracy
    fig, ax = plt.subplots(figsize=(10, 5.3))
    for name, d in data.items():
        band_line(ax, np.arange(d["n"]), d["acc"], colors[name], name)
    ax.set_title(r"Final test accuracy $\mathrm{Acc}(t)$")
    ax.set_xlabel(r"task index $t$"); ax.set_ylabel(r"test accuracy")
    finish(fig, [ax], None, "drift_story_1_accuracy.png", ncol=2)

    # 2 alive units
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.3), sharey=True)
    for name, d in data.items():
        x = np.arange(d["n"])
        band_line(axes[0], x, d["A1"], colors[name], name)
        band_line(axes[1], x, d["A2"], colors[name], name)
    for ax, k in zip(axes, (1, 2)):
        ax.set_title(rf"Layer ${k}$: alive units $A_{k}(t)$")
        ax.set_xlabel(r"task index $t$"); ax.set_ylabel(r"alive units (of $30$)")
    finish(fig, axes, None, "drift_story_2_alive.png", ncol=2)

    # 3 turnover
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.3), sharey=True)
    names = {"R": r"$R$: revived at the switch", "K": r"$K$: killed at the switch", "O": r"$O$: killed by optimization"}
    for ax, (name, d) in zip(axes, data.items()):
        x = np.arange(d["n"])
        for i, q in enumerate("RKO"):
            band_line(ax, x, d[q], common.style_for(i), names[q])
        ax.set_title(rf"{name}, layer $1$")
        ax.set_xlabel(r"task index $t$"); ax.set_ylabel(r"units per task")
    finish(fig, axes, None, "drift_story_3_turnover.png", ncol=3)

    # 4 gradient, input drift
    name, d = next(iter(data.items()))
    x = np.arange(d["n"])
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.3))
    band_line(axes[0], x, d["g2"], common.style_for(0), r"$g_2$: gradient on layer-2 weights", log=True)
    band_line(axes[0], x, d["g1"], common.style_for(1), r"$g_1$: gradient on layer-1 weights", log=True)
    axes[0].set_title(r"Gradient at the start of each task")
    axes[0].set_ylabel(r"gradient norm")
    band_line(axes[1], x, d["alive_probe"], common.style_for(0), r"alive layer-1 units")
    band_line(axes[1], x, d["reach"], common.style_for(1), rf"reached by the gradient ($> {REACH:g} \times$ task-0 median)")
    band_line(axes[1], x, d["partial"], common.style_for(2), r"partially active ($5$--$95\%$ of $D_t$)")
    axes[1].set_title(r"Layer-$1$ units")
    axes[1].set_ylabel(r"units (of $30$)")
    for ax in axes:
        marker(ax, d["T_2"], "0.2", r"$T_2$: layer-2 death")
        ax.set_xlabel(r"task index $t$")
    finish(fig, axes, rf"{name}: layer-1 units stay alive but stop receiving gradient",
           "drift_story_4_gradient.png", ncol=3)


if __name__ == "__main__":
    main()
