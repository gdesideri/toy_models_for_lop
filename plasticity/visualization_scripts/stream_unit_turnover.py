"""
visualization_scripts/stream_unit_turnover.py — Where hidden units die and revive in a
task stream: at the switch between tasks, or during training on a task.

Per task t and layer (a unit is active if its output is > 0 on at least one probe image):
    R_t  revived at the switch   dead at the end of task t-1 on D_{t-1},
                                 active with the same weights on D_t
    K_t  killed at the switch    active at the end of task t-1 on D_{t-1},
                                 dead with the same weights on D_t
    O_t  killed by training      active at the start of task t on D_t,
                                 dead at the end of task t (10 epochs) on D_t
The start of task t is the end of task t-1 (the initial network for task 0), so R and K
isolate the change of distribution and O the training on task t. In the feature-switch
stream consecutive tasks share their images, so R = K = 0 there by construction.

Probe: N_PROBE images drawn once per seed from the run's own training split; for each task
they are shown as that task presents them (its pixel permutation).

Measuring runs on a GPU, one (stream, seed) per call (experiments/launch_scripts/submit_unit_turnover.sh):
    uv run python -m visualization_scripts.stream_unit_turnover --compute <stream dir> --seed <k> [--n_probe N]
Plotting reads the per-seed files:
    uv run python -m visualization_scripts.stream_unit_turnover [layer] [--n_probe N]
Writes   : <stream dir>/unit_turnover_n<N>_seed<k>.json, results/stream_unit_turnover_n<N>_layer<l>.png
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.checkpoints import load_checkpoint
from utils.evaluation import alive_masks_over
from utils.stream_probe import (draw_probe, find_run, load_init_model, task_batches, task_checkpoints,
                                task_dirs, task_perm, task_training_images)

from visualization_scripts import common

N_PROBE = 20_000
PROBE_SALT = 29                     # also the probe of stream_killed_units_width.py
SMOOTH = 25


def compute_seed(stream_dir, seed, device, n_probe=N_PROBE):
    run = find_run(stream_dir, seed)
    t0 = time.time()
    _, images = task_training_images(run, device)
    _, probe = draw_probe(images, seed, n_probe, PROBE_SALT)
    n_probe = len(probe)
    del images
    tasks = task_dirs(run)
    perms = [task_perm(t) for t in tasks]
    masks = lambda m, p: alive_masks_over(m, task_batches(probe, p))

    model = load_init_model(run, device)
    out = {"measured_on": f"{n_probe} images of the run's training split, task permutation applied",
           "n_probe": n_probe, "task": [], "R": {}, "K": {}, "O": {}, "revived_by_training": {}}
    end_prev = None                                            # end of task t-1 on D_{t-1}
    for n, t in enumerate(tasks):
        start = masks(model, perms[n])                         # start of task t on D_t (weights of t-1's end)
        load_checkpoint(task_checkpoints(t)[-1], model)
        end = masks(model, perms[n])                           # end of task t on D_t
        out["task"].append(n)
        for k in range(len(start)):
            key = f"layer_{k + 1}"
            if end_prev is None:
                r = kk = None
            else:
                r, kk = int((~end_prev[k] & start[k]).sum()), int((end_prev[k] & ~start[k]).sum())
            out["R"].setdefault(key, []).append(r)
            out["K"].setdefault(key, []).append(kk)
            out["O"].setdefault(key, []).append(int((start[k] & ~end[k]).sum()))
            out["revived_by_training"].setdefault(key, []).append(int((~start[k] & end[k]).sum()))
        end_prev = end
    json.dump(out, open(Path(stream_dir) / f"unit_turnover_n{n_probe}_seed{seed}.json", "w"))
    print(f"{run.name}: {len(tasks)} tasks in {time.time() - t0:.0f}s")


def main(layer=1, n_probe=N_PROBE):
    streams = common.task_streams()
    common.set_style()
    fig, axes = plt.subplots(1, len(streams), figsize=(8.5 * len(streams), 5.4), squeeze=False, sharey=True)
    names = {"R": r"$R$: revived at the switch", "K": r"$K$: killed at the switch", "O": r"$O$: killed during training"}
    for ax, (stream_dir, title) in zip(axes[0], streams):
        files = sorted(glob.glob(str(stream_dir / f"unit_turnover_n{n_probe}_seed*.json")))
        if not files:
            raise SystemExit(f"no unit_turnover_n{n_probe}_seed*.json in {stream_dir}: "
                             f"run N_PROBE={n_probe} experiments/launch_scripts/submit_unit_turnover.sh")
        runs = [json.load(open(f)) for f in files]
        n = min(len(r["task"]) for r in runs)
        tasks = np.arange(n)
        for i, q in enumerate(("R", "K", "O")):
            y = np.nanmean(np.array([[np.nan if v is None else v for v in r[q][f"layer_{layer}"][:n]] for r in runs],
                                    dtype=float), axis=0)
            style = common.style_for(i)
            ax.plot(tasks, y, color=style["color"], linestyle="-", linewidth=0.6, alpha=0.3)
            ax.plot(tasks, common.moving_average(y, SMOOTH), **style, linewidth=2.0, label=names[q])
        ax.set_title(title + rf" ($n = {len(runs)}$ seeds)")
        ax.set_xlabel(r"task index $t$")
        ax.set_ylabel(r"units per task")
    common.bold_suptitle(fig, rf"Layer-${layer}$ units changing state, per task (mean over seeds; "
                              rf"thick: ${SMOOTH}$-task average; probe: ${n_probe // 1000}$k training images per task)")
    common.legend_below(fig, *axes[0][0].get_legend_handles_labels(), bottom=0.08, ncol=3)
    common.save(fig, common.RESULTS / f"stream_unit_turnover_n{n_probe}_layer{layer}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("layer", nargs="?", type=int, default=1)
    p.add_argument("--compute", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_probe", type=int, default=N_PROBE, help="probe images per seed")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.compute:
        compute_seed(a.compute, a.seed, a.device, a.n_probe)
    else:
        main(a.layer, a.n_probe)
