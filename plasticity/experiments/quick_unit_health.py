"""
experiments/quick_unit_health.py — Fast local PILOT of a cluster-factor stream under
several input / training conditions, measuring the hidden units after every task.
For deciding what to submit, never for final numbers (use run_tasks_stream for those).

The data, the cluster-factor labels (6 factors x 8 clusters, 50 PCA dims) and the
training set live in memory (on mps / cuda when available), so a 12-task x 40-epoch
stream of an mlp_2x30 takes about a minute; Shapes3D's labels take ~6 min to fit,
once per call. Same model, optimiser (SGD, momentum 0), batch, parity labels and
carried-over head as run_tasks_stream; shuffling differs.

    uv run python -m experiments.quick_unit_health --dataset shapes3d --conditions base center=1 lr=0.0001
    uv run python -m experiments.quick_unit_health --dataset cifar10 --conditions base offset=0.9113 lr=0.01

A condition is `base` or comma-separated settings: center, offset, lr, model, activation
(e.g. `center=0.5,model=mlp_2x120`). Prints, per task: train accuracy, dead and rarely
active units per layer, median activation rate of each layer, effective rank, |W1|.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.cluster_factors import cluster_cache_path, with_cluster_factors
from data.factory import build_dataset
from data.input_shift import input_stats, offset_for_ratio  # noqa: F401  (offset_for_ratio: handy in a REPL)
from models.factory import build_model
from utils.evaluation import unit_health

DEFAULTS = dict(center=0.0, offset=0.0, lr=1e-3, model="mlp_2x30", activation="relu")


def parse_condition(text):
    cond = dict(DEFAULTS)
    if text != "base":
        for item in text.split(","):
            key, value = item.split("=")
            if key not in cond:
                raise SystemExit(f"unknown setting {key!r}; use {list(cond)}")
            cond[key] = type(DEFAULTS[key])(value)
    return cond


def load(kind, n_samples):
    if kind == "shapes3d":
        train, test = build_dataset("shapes3d", n_samples=n_samples, rng=np.random.default_rng(0))
        cache = None
    else:
        train, test = build_dataset(kind)
        cache = cluster_cache_path(kind, 6, 8, 50)
    factors, _ = with_cluster_factors(train, test, 6, 8, 50, cache_path=cache)
    with torch.random.fork_rng(devices=[]):
        x = torch.cat([xb for xb, _ in DataLoader(train, batch_size=2000)]).reshape(len(train), -1)
    y = torch.from_numpy((factors.latent_idx % 2 == 0).astype(np.int64))
    return x.float(), y


def run(x, y, cond, tasks, epochs, batch, device, seed):
    shift = (-cond["center"] * x.mean(0) + cond["offset"]).to(device)
    stats = input_stats(TensorDataset(x + shift.cpu(), torch.zeros(len(x))))
    xd, yd = x.to(device), y.to(device)
    probe = xd[torch.from_numpy(np.random.default_rng(seed).choice(len(x), min(5000, len(x)), replace=False)).to(device)]
    torch.manual_seed(seed)
    model = build_model(cond["model"], d_in=x.shape[1], n_classes=2, activation=cond["activation"]).to(device)
    print(f"\n== {cond} | push/spread {stats['push_over_spread']:.1f} | x lr {stats['push_over_spread'] * cond['lr']:.3g}")
    print(f"{'task':>4} {'fac':>3} {'train':>6} {'dead L1/L2':>10} {'rare L1/L2':>10} {'median L1/L2':>13} "
          f"{'eRank':>5} {'|W1|':>5}")
    g = torch.Generator().manual_seed(seed)

    def report(t, f, acc):
        h = unit_health(model, ((xd[i:i + 8192] + shift) for i in range(0, len(xd), 8192)), probe + shift)
        l1, l2 = h["layers"]
        print(f"{t:>4} {f:>3} {acc:>6} {l1['dead']:>4}/{l2['dead']:<5} {l1['rare']:>4}/{l2['rare']:<5} "
              f"{l1['median_rate']:>6.2f}/{l2['median_rate']:<6.2f} {h['effective_rank']:>5.1f} "
              f"{h['weight_norms'][0]:>5.1f}", flush=True)

    report("init", "", "")
    for t in range(tasks):
        f = t % y.shape[1]
        opt = torch.optim.SGD(model.parameters(), lr=cond["lr"])
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(xd), generator=g).to(device)
            for i in range(0, len(xd), batch):
                b = perm[i:i + batch]
                loss = F.cross_entropy(model(xd[b] + shift), yd[b, f])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = torch.cat([model(xd[i:i + 8192] + shift).argmax(1) for i in range(0, len(xd), 8192)])
            acc = f"{float(pred.eq(yd[:, f]).float().mean()):.3f}"
        report(t, f, acc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10", "shapes3d"])
    p.add_argument("--conditions", nargs="+", default=["base"])
    p.add_argument("--tasks", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n_samples", type=int, default=50000, help="shapes3d only")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    x, y = load(a.dataset, a.n_samples)
    print(f"{a.dataset}: {x.shape[0]} images x {x.shape[1]} inputs, loaded in {time.time() - t0:.0f}s on {device}")
    for text in a.conditions:
        run(x, y, parse_condition(text), a.tasks, a.epochs, a.batch, device, a.seed)


if __name__ == "__main__":
    main()
