"""
visualization_scripts/stream_unit_transitions.py — Per task, how many hidden units
revive (dead at the end of task t-1, alive at the end of task t) and how many die
(alive at the end of t-1, dead at the end of t), for both stream families and both
layers. Task 0 is compared with the initial network.

Alive/dead as in stream_alive_units.py: > 0 on at least one probe image, shown as
the task presents them. In the permutation stream a unit can therefore also change
state because task t's permutation feeds it different inputs than task t-1's.

Thin lines: mean over seeds per task. Thick lines: that mean smoothed over a
window of SMOOTH tasks.

Run with:  uv run python -m visualization_scripts.stream_unit_transitions
Reads    : <stream dir>/alive_units_seed<k>.json (measured by stream_alive_units.py --compute)
Writes   : results/stream_unit_transitions.png
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization_scripts import common
from visualization_scripts.stream_alive_units import load_alive_data

SMOOTH = 25


def transitions(run, k):
    """(revived, died) per task for layer k of one seed."""
    masks = np.array([[c == "1" for c in m] for m in [run[f"init_mask_layer_{k}"]] + run[f"mask_layer_{k}"]])
    prev, cur = masks[:-1], masks[1:]
    return (~prev & cur).sum(1), (prev & ~cur).sum(1)


def main():
    data = {title: load_alive_data(stream_dir) for stream_dir, title in common.task_streams()}
    n_layers = max(len([k for k in v if k.startswith("mask_layer_")]) for d in data.values() for v in d.values())
    common.set_style()
    fig, axes = plt.subplots(len(data), n_layers, figsize=(8 * n_layers, 4.6 * len(data)),
                             squeeze=False, sharey=True)
    for row, (title, runs) in enumerate(data.items()):
        n = min(len(r["task"]) for r in runs.values())
        tasks = np.arange(n)
        for k in range(1, n_layers + 1):
            ax = axes[row][k - 1]
            rev, died = zip(*(transitions(r, k) for r in runs.values()))
            for i, (name, y) in enumerate([(r"revived (dead $\rightarrow$ alive)", np.mean([v[:n] for v in rev], 0)),
                                           (r"died (alive $\rightarrow$ dead)", np.mean([v[:n] for v in died], 0))]):
                style = common.style_for(i)
                ax.plot(tasks, y, color=style["color"], linestyle="-", linewidth=0.6, alpha=0.35)
                ax.plot(tasks, common.smooth(y, SMOOTH), **style, linewidth=2.0, label=name)
            ax.set_title(title + rf", layer ${k}$")
            ax.set_xlabel(r"task index")
            ax.set_ylabel(r"units per task")
    common.bold_suptitle(fig, r"Units changing state between the end of task $t-1$ and the end of task $t$ "
                                 rf"(mean over seeds; thick: ${SMOOTH}$-task average)")
    common.legend_below(fig, *axes[0][0].get_legend_handles_labels(), bottom=0.05)
    common.save(fig, common.RESULTS / "stream_unit_transitions.png")


if __name__ == "__main__":
    main()
