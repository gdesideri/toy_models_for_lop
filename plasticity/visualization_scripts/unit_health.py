"""
visualization_scripts/unit_health.py — Hidden-unit health over task streams: dead and
rarely active units per layer against the task index, and how fast units die against
the shared push of the inputs times the learning rate (the recap doc's E1-E4).

Runs that logged their health (run_tasks_stream --health_every k, the default) carry it in
summary.json. Older runs are measured once from their end-of-task checkpoints, on the same
data and with the same probe as the logging (utils.evaluation.unit_health):
    uv run python -m visualization_scripts.unit_health --compute <stream dir> --seed 0 [--every 5]
writes <run dir>/unit_health.json (input_stats, init_health, per measured task: health);
experiments/launch_scripts/submit_unit_health.sh runs it on the cluster.

Compare named stream directories (each holding stream_*_seed* runs, one per seed):
    uv run python -m visualization_scripts.unit_health --name e1_shapes3d \
        --runs baseline=results/shapes3d_cluster_stream_mlp_2x30 centred=results/e1_shapes3d_cluster_c1_mlp_2x30
Plots, per group (mean over seeds, band = min-max), dead units of each layer against the
number of tasks done, and death speed (mean dead fraction over the first 200 tasks) against
push/spread x lr; prints dead units at tasks 50 / 200 / last and accuracy of the first and
last 50 tasks. Writes results/unit_health_<name>.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.factory import build_dataset, build_eval_loader
from data.input_shift import ShiftedInputs, input_stats, shift_for
from tasks.permuted_pixels import permute_dataset
from utils.checkpoints import load_checkpoint
from utils.evaluation import unit_health
from utils.stream_probe import find_run, load_init_model, task_checkpoints, task_dirs, task_perm

SPEED_TASKS = 200          # death speed = mean dead fraction over the first this many tasks


# ── measuring old runs ────────────────────────────────────────────────────────

def compute(stream_dir, seed, every, device, workers=0):
    """unit_health of a run's end-of-task checkpoints (every `every`-th task and the last),
    on the run's training set as the stream saw it; written to <run>/unit_health.json."""
    run = find_run(stream_dir, seed)
    a = json.load(open(run / "args.json"))
    train, _ = build_dataset(a["dataset_kind"], n_samples=a.get("n_samples"), target_factor=a.get("target_factor"),
                             rng=np.random.default_rng(a["seed"]), label_rule=a.get("label_rule"))
    shift = shift_for(train, a.get("center_inputs", 0.0), a.get("input_offset", 0.0))
    if shift is not None:
        train = ShiftedInputs(train, shift)
    probe_idx = np.sort(np.random.default_rng([a["seed"], 11]).choice(
        len(train), min(5000, len(train)), replace=False))
    model = load_init_model(run, device)
    n_pixels = int(np.prod(train[0][0].shape[-2:]))

    def health(perm):
        data = permute_dataset(train, perm)
        with torch.random.fork_rng(devices=[]):
            batches = (x.to(device) for x, _ in build_eval_loader(data, 2048, workers))
            probe = torch.stack([data[int(i)][0] for i in probe_idx]).to(device)
            return unit_health(model, batches, probe)

    out = {"every": every, "input_stats": input_stats(train), "init_health": health(np.arange(n_pixels)),
           "tasks": []}
    tasks = task_dirs(run)
    for i, t in enumerate(tasks):
        if (i + 1) % every and i != len(tasks) - 1:
            continue
        load_checkpoint(task_checkpoints(t)[-1], model)
        out["tasks"].append({"task": int(t.name.split("_")[1]), "health": health(task_perm(t).numpy())})
        print(f"  task {out['tasks'][-1]['task']:4d}: dead {[l['dead'] for l in out['tasks'][-1]['health']['layers']]}",
              flush=True)
    json.dump(out, open(run / "unit_health.json", "w"))
    print(f"-> {run / 'unit_health.json'}")
    return out


# ── reading ───────────────────────────────────────────────────────────────────

def load_group(stream_dir):
    """Per run (seed) of a stream directory: args, tasks, input_stats and the health series."""
    runs = []
    for run in sorted(Path(stream_dir).glob("stream_*_seed*")):
        a = json.load(open(run / "args.json"))
        s = json.load(open(run / "summary.json"))
        if any("health" in t for t in s["tasks"]):
            series = [(t["task"], t["health"]) for t in s["tasks"] if "health" in t]
            stats = s.get("input_stats")
        elif (run / "unit_health.json").exists():
            u = json.load(open(run / "unit_health.json"))
            series, stats = [(t["task"], t["health"]) for t in u["tasks"]], u["input_stats"]
        else:
            print(f"WARNING: {run} has no unit health (run --compute first)")
            series, stats = [], None
        runs.append(dict(name=run.name, args=a, tasks=s["tasks"], stats=stats, series=series))
    if not runs:
        raise SystemExit(f"no stream_*_seed* runs in {stream_dir}")
    return runs


def dead_by_task(run, layer):
    return {t + 1: h["layers"][layer]["dead"] for t, h in run["series"]}       # tasks done -> dead units


def death_speed(run):
    """Mean fraction of all hidden units dead, over the measured tasks among the first SPEED_TASKS."""
    pts = [sum(l["dead"] for l in h["layers"]) / sum(l["units"] for l in h["layers"])
           for t, h in run["series"] if t < SPEED_TASKS]
    return float(np.mean(pts)) if pts else float("nan")


def push_times_lr(run):
    return run["stats"]["push_over_spread"] * run["args"]["learning_rate"] if run["stats"] else float("nan")


def at(series, task):
    """The value at the measured task closest to `task` (tasks done), or nan."""
    if not series:
        return float("nan")
    k = min(series, key=lambda t: abs(t - task))
    return series[k]


def acc(run, which, first):
    ts = run["tasks"][:50] if first else run["tasks"][-50:]
    return float(np.mean([t[f"final_{which}_acc"] for t in ts])) if ts else float("nan")


# ── report ────────────────────────────────────────────────────────────────────

def report(groups, name):
    from visualization_scripts import common
    print(f"{'group':28s} {'runs':>4} {'push x lr':>9} {'L1 dead 50/200/last':>20} {'L2 dead 50/200/last':>20} "
          f"{'test 1-50 > last 50':>20} {'train 1-50 > last 50':>21}")
    for g, runs in groups.items():
        cells = []
        for layer in (0, 1):
            vals = [[at(dead_by_task(r, layer), k) for r in runs] for k in (50, 200, 10 ** 9)]
            cells.append("/".join(f"{np.nanmean(v):.1f}" for v in vals))
        accs = [f"{np.nanmean([acc(r, w, True) for r in runs]):.3f} > {np.nanmean([acc(r, w, False) for r in runs]):.3f}"
                for w in ("test", "train")]
        print(f"{g:28s} {len(runs):>4} {np.nanmean([push_times_lr(r) for r in runs]):>9.3g} {cells[0]:>20} "
              f"{cells[1]:>20} {accs[0]:>20} {accs[1]:>21}")

    common.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.2))
    for i, (g, runs) in enumerate(groups.items()):
        style = common.style_for(i)
        for layer in (0, 1):
            series = [dead_by_task(r, layer) for r in runs if r["series"]]
            if not series:
                continue
            common_tasks = sorted(set.intersection(*(set(s) for s in series)))
            rows = [[s[t] for t in common_tasks] for s in series]
            common.band(axes[layer], common_tasks, rows, style["color"], label=f"{g} ($n={len(rows)}$)",
                        linestyle=style["linestyle"], linewidth=1.5)
        x = [max(push_times_lr(r), 1e-3) for r in runs]
        y = [death_speed(r) for r in runs]
        axes[2].scatter(x, y, color=style["color"], s=40, zorder=3)
        axes[2].scatter([np.mean(x)], [np.nanmean(y)], color=style["color"], marker="D", s=90, zorder=4)
    for layer in (0, 1):
        axes[layer].set_title(rf"Layer ${layer + 1}$: dead units")
        axes[layer].set_xlabel("tasks done")
        axes[layer].set_ylabel("dead units")
    axes[2].set_xscale("log")
    axes[2].set_title(rf"Death speed (mean dead fraction, first {SPEED_TASKS} tasks)")
    axes[2].set_xlabel(r"shared push / spread $\times$ lr  (centred runs at $10^{-3}$)")
    axes[2].set_ylabel("mean dead fraction")
    common.legend_below(fig, *axes[0].get_legend_handles_labels(), bottom=0.12)
    common.save(fig, common.RESULTS / f"unit_health_{name}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compute", type=str, default=None, help="stream results directory to measure")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--every", type=int, default=5, help="--compute: measure every this many tasks (and the last)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--runs", nargs="*", default=[], help="name=stream_dir pairs to compare")
    p.add_argument("--name", type=str, default="comparison")
    a = p.parse_args()
    if a.compute:
        compute(a.compute, a.seed, a.every, a.device, a.workers)
    elif a.runs:
        report({n: load_group(d) for n, d in (r.split("=", 1) for r in a.runs)}, a.name)
    else:
        p.error("give --compute <stream dir> --seed k, or --runs name=dir ...")


if __name__ == "__main__":
    main()
