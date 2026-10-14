"""
visualization_scripts/stream_alive_units.py — Alive ReLU units per hidden layer
over a task stream (experiments/run_tasks_stream.py): one panel per layer, one
line per stream (mean over seeds, band = min-max over seeds).

A unit is alive at the end of task t if its ReLU output is > 0 for at least one
image of task t's own training data -- the seed's training split, rebuilt exactly
as the stream built it, normalised as in training and with task t's pixel
permutation. Dead = 0 on every one of them. The network is the task's end-of-task
snapshot; the initial network is measured on task 0's data.

Measuring is a forward pass over 432 000 images per checkpoint, so it runs on a
GPU, one (stream, seed) per call; experiments/launch_scripts/submit_alive_units.sh
runs all of them as a Slurm array:
    uv run python -m visualization_scripts.stream_alive_units --compute <stream dir> --seed <k>
Plotting reads the per-seed caches:
    uv run python -m visualization_scripts.stream_alive_units

Writes   : <stream dir>/alive_units_seed<k>.json   task index, alive units per layer and the
                                                    per-unit masks ("1" = alive), incl. init
           results/stream_alive_units.png
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
from utils.stream_probe import (find_run, load_init_model, task_batches, task_checkpoints, task_dirs,
                                task_perm, task_training_images)

from visualization_scripts import common

MEASURED_ON = "task training split"


# ── measuring ─────────────────────────────────────────────────────────────────

def _bits(mask):
    return "".join("1" if a else "0" for a in mask.tolist())


def compute_seed(stream_dir, seed, device):
    """Masks and counts of every task of one seed of one stream, cached to alive_units_seed<k>.json."""
    run = find_run(stream_dir, seed)
    t0 = time.time()
    train, images = task_training_images(run, device)
    print(f"{run.name}: {len(images)} training images on {device} ({time.time() - t0:.0f}s)")

    # the batched transform must reproduce what training saw (PermutedPixels over the dataset)
    from tasks.permuted_pixels import PermutedPixels
    tasks = task_dirs(run)
    perm0 = task_perm(tasks[0])
    ref = torch.stack([PermutedPixels(train, perm0.numpy())[i][0] for i in range(8)])
    mine = next(task_batches(images[:8], perm0)).cpu()
    assert torch.allclose(ref, mine, atol=1e-6), "batched task inputs differ from the training inputs"

    model = load_init_model(run, device)
    rows = {"measured_on": MEASURED_ON, "n_images": len(images), "task": []}
    for k, m in enumerate(alive_masks_over(model, task_batches(images, perm0)), start=1):
        rows[f"init_mask_layer_{k}"] = _bits(m)
    for t in tasks:
        load_checkpoint(task_checkpoints(t)[-1], model)
        masks = alive_masks_over(model, task_batches(images, task_perm(t)))
        rows["task"].append(int(t.name.split("_")[1]))
        for k, m in enumerate(masks, start=1):
            rows.setdefault(f"layer_{k}", []).append(int(m.sum()))
            rows.setdefault(f"mask_layer_{k}", []).append(_bits(m))
    json.dump(rows, open(Path(stream_dir) / f"alive_units_seed{seed}.json", "w"))
    print(f"  {len(tasks)} tasks in {time.time() - t0:.0f}s; alive at the end: "
          f"{[rows[k][-1] for k in rows if k.startswith('layer_')]}")
    return rows


def load_alive_data(stream_dir):
    """{seed: rows} for every seed of a stream, from the per-seed caches measured on task data."""
    stream_dir = Path(stream_dir)
    seeds = sorted(r.rsplit("seed", 1)[1] for r in glob.glob(str(stream_dir / "stream_*_seed*")))
    data = {}
    for s in seeds:
        f = stream_dir / f"alive_units_seed{s}.json"
        if not f.exists():
            raise SystemExit(f"{f} missing: run experiments/launch_scripts/submit_alive_units.sh (GPU) first")
        rows = json.load(open(f))
        if rows.get("measured_on") != MEASURED_ON:
            raise SystemExit(f"{f} was not measured on the task data")
        data[s] = rows
    return data


# ── plotting ──────────────────────────────────────────────────────────────────

def main():
    data = {title: load_alive_data(stream_dir) for stream_dir, title in common.task_streams()}
    n_layers = max(len([k for k in v if k.startswith("layer_")]) for d in data.values() for v in d.values())
    common.set_style()
    fig, axes = plt.subplots(1, n_layers, figsize=(8 * n_layers, 5.2), squeeze=False, sharey=True)
    for k in range(1, n_layers + 1):
        ax = axes[0][k - 1]
        for i, (title, runs) in enumerate(data.items()):
            n = min(len(r["task"]) for r in runs.values())
            tasks = np.array(next(iter(runs.values()))["task"][:n])
            counts = np.array([r[f"layer_{k}"][:n] for r in runs.values()], dtype=float)   # (seeds, tasks)
            common.band(ax, tasks, counts, **common.style_for(i), linewidth=1.3,
                        label=title + rf" ($n = {len(runs)}$ seeds)")
        ax.set_title(rf"Layer ${k}$")
        ax.set_xlabel(r"task index")
        ax.set_ylabel(r"alive units")
    common.bold_suptitle(fig, r"Alive ReLU units at the end of each task, on the task's training data, "
                              r"MLP $2 \times 30$, SGD (mean and range over seeds)")
    common.legend_below(fig, *axes[0][0].get_legend_handles_labels(), bottom=0.08)
    common.save(fig, common.RESULTS / "stream_alive_units.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--compute", type=str, default=None, help="stream results dir to measure")
    p.add_argument("--seed", type=int, default=None, help="seed of that stream to measure")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.compute:
        compute_seed(a.compute, a.seed, a.device)
    else:
        main()
