"""
visualization_scripts/stream_gradient_reach.py — How much gradient reaches each layer at
the start of every task of a task stream, and what layer 1 looks like at its end.

Per task t (probe: N_PROBE images of the run's training split, shown as task t presents
them -- its pixel permutation -- with task t's labels):
    g_1, g_2        size (Frobenius norm) of the loss gradient w.r.t. the layer-1 / layer-2
                    weights at the start of task t (= the weights at the end of task t-1)
    unit_grad_1     per layer-1 unit, the norm of the gradient of its weights and bias
                    (a layer-1 unit only receives gradient through the alive layer-2 units)
    rates_1         per layer-1 unit, the fraction of the probe it is active on at the end of task t
    dW_1            how far the layer-1 weights move during task t (norm of end - start)

Runs on a GPU, one (stream, seed) per call (experiments/launch_scripts/submit_gradient_reach.sh):
    uv run python -m visualization_scripts.stream_gradient_reach --compute <stream dir> --seed <k>
Writes   : <stream dir>/gradient_reach_n<N>_seed<k>.json
The charts are drawn by visualization_scripts/drift_story.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tasks.feature_streams import FeatureParityTask
from utils.checkpoints import load_checkpoint
from utils.evaluation import activation_rates_over
from utils.stream_probe import (draw_probe, find_run, load_init_model, task_batches, task_checkpoints,
                                task_dirs, task_perm, task_training_images)

N_PROBE = 100_000
PROBE_SALT = 31


def compute_seed(stream_dir, seed, device, n_probe=N_PROBE, batch=8192):
    run = find_run(stream_dir, seed)
    t0 = time.time()
    a = json.load(open(run / "args.json"))
    train, images = task_training_images(run, device)
    idx, probe = draw_probe(images, seed, n_probe, PROBE_SALT)
    n_probe = len(probe)
    latent = np.asarray(train.latent_idx)[idx]
    del images
    tasks = task_dirs(run)
    summary = json.load(open(run / "summary.json"))["tasks"]
    factor = lambda n: summary[n].get("target_factor", a["target_factor"])   # label feature of task n
    labels = {f: torch.from_numpy(FeatureParityTask([f]).labels(latent)).to(device) for f in range(6)}

    model = load_init_model(run, device)
    lin1, lin2 = model.body[0], model.body[2]
    ce = nn.CrossEntropyLoss(reduction="sum")
    out = {"measured_on": f"{n_probe} images of the run's training split, task permutation and labels",
           "n_probe": n_probe, "task": [], "g_1": [], "g_2": [], "unit_grad_1": [], "rates_1": [], "dW_1": []}
    for n, t in enumerate(tasks):
        perm = task_perm(t)
        y = labels[factor(n)]
        # gradient at the start of task t, on D_t with task t's labels
        model.zero_grad()
        model.eval()
        for i, xb in enumerate(task_batches(probe, perm, batch_size=batch)):
            (ce(model(xb), y[i * batch:(i + 1) * batch]) / n_probe).backward()
        g1w, g1b = lin1.weight.grad, lin1.bias.grad
        out["g_1"].append(float(g1w.norm()))
        out["g_2"].append(float(lin2.weight.grad.norm()))
        out["unit_grad_1"].append(torch.sqrt((g1w ** 2).sum(1) + g1b ** 2).tolist())
        w_start = lin1.weight.detach().clone()
        # the end of task t
        load_checkpoint(task_checkpoints(t)[-1], model)
        out["dW_1"].append(float((lin1.weight.detach() - w_start).norm()))
        out["rates_1"].append(activation_rates_over(model, task_batches(probe, perm))[0].tolist())
        out["task"].append(n)
    json.dump(out, open(Path(stream_dir) / f"gradient_reach_n{n_probe}_seed{seed}.json", "w"))
    print(f"{run.name}: {len(tasks)} tasks in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--compute", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_probe", type=int, default=N_PROBE)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    compute_seed(a.compute, a.seed, a.device, a.n_probe)
