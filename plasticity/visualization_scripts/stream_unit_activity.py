"""
visualization_scripts/stream_unit_activity.py — Activity of every hidden unit of one
layer over the epochs of a window of consecutive tasks, one row per start task.

y = % of the current task's own training data on which the unit is active (output > 0):
during the epochs of task n it is measured on D_n only. 0% = dead on that task's data.
Switches between tasks are marked by solid lines and alternating shaded bands.
Data from experiments/replay_stream_unit_activity.py (run it first, on a GPU).

Run with:  uv run python -m visualization_scripts.stream_unit_activity [run dir|-] [layer] [window]
           (default: the permuted-pixel stream, seed 0; layer 1; window 5 tasks)
Writes   : results/stream_unit_activity_<stream>_seed<k>_layer<l>_w<window>.png
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from utils.stream_probe import find_run
from visualization_scripts import common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="-", help="stream run dir ('-': the permuted-pixel stream, seed 0)")
    p.add_argument("layer", nargs="?", type=int, default=1)
    p.add_argument("window", nargs="?", type=int, default=5, help="tasks replayed from each start task")
    args = p.parse_args()
    run = find_run(common.stream_dir(common.PERM_STREAM), 0) if args.run == "-" else Path(args.run).resolve()
    layer, window = args.layer, args.window
    files = sorted(glob.glob(str(run / f"unit_activity_t*_w{window}.json")),
                   key=lambda f: int(re.search(r"_t(\d+)_", f).group(1)))
    if not files:
        raise SystemExit(f"no unit_activity_t*_w{window}.json in {run}: run "
                         f"WINDOW={window} experiments/launch_scripts/submit_replay_unit_activity.sh first")
    data = [json.load(open(f)) for f in files]

    common.set_style()
    fig, axes = plt.subplots(len(data), 1, figsize=(17, 4.4 * len(data)), squeeze=False, sharey=True)
    for ax, d in zip(axes[:, 0], data):
        rates = 100 * np.array(d["rates"][f"layer_{layer}"])          # (epochs, units)
        x = np.array(d["epochs"])
        task = np.array(d["task"])
        n_units = rates.shape[1]
        norm = Normalize(0, n_units - 1)
        # one shaded band per task, a solid line at every switch, the task and its data on top
        for i, n in enumerate(sorted(set(task))):
            lo, hi = x[task == n].min() - 0.5, x[task == n].max() + 0.5
            if i % 2 == 0:
                ax.axvspan(lo, hi, color="0.9", zorder=0)
            if i > 0:
                ax.axvline(lo, color="black", linewidth=2.0, zorder=3)
            ax.text((lo + hi) / 2, 104, rf"task ${n}$ on $D_{{{n}}}$", ha="center", va="bottom", fontsize=13)
        # each task's segment is drawn on its own: points of different tasks are measured on
        # different distributions, so the lines are not joined across a switch
        for u in range(n_units):
            for n in sorted(set(task)):
                m = task == n
                ax.plot(x[m], rates[m, u], color=common.CMAP(norm(u)),
                        linestyle=common.LINESTYLES[u % len(common.LINESTYLES)], linewidth=1.1, zorder=2)
        epochs_per_task = int((task == task[0]).sum())
        within = (x - 1) % epochs_per_task + 1                                  # epoch within its task
        show = np.isin(within, [1, 5, epochs_per_task])
        ax.set_xticks(x[show], [str(e) for e in within[show]])
        ax.set_xticks(x, minor=True)
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(-3, 103)
        ax.set_xlabel(r"epoch within each task")
        ax.set_ylabel(r"\% of $D_n$ active" if plt.rcParams["text.usetex"] else r"% of $D_n$ active")
        ax.set_title(rf"Tasks ${d['t']}$ to ${d['t'] + len(set(task)) - 1}$", pad=26)
    seed = re.search(r"seed(\d+)", run.name).group(1)
    common.bold_suptitle(fig, rf"Layer-${layer}$ units ($n = {n_units}$): % of each task's own training data "
                                 rf"on which the unit is active, permuted-pixel stream, MLP $2 \times 30$, seed ${seed}$")
    fig.tight_layout(rect=(0, 0, 0.94, 0.975), h_pad=2.5)
    cax = fig.add_axes([0.952, 0.2, 0.01, 0.6])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=common.CMAP), cax=cax)
    cb.set_label(r"unit index")
    common.save(fig, common.RESULTS / f"stream_unit_activity_{run.parent.name}_seed{seed}_layer{layer}_w{window}.png")


if __name__ == "__main__":
    main()
