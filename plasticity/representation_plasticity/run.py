"""
representation_plasticity/run.py — Experiment runner for the representation setting.

Run:
    uv run python -m representation_plasticity.run \
        --stream ladder --arch mlp_2x512 --activation relu \
        --switch_every 2000 --n_cycles 2

The script loads Shapes3D from the data cache (must be downloaded first via
    from data.download import get_shapes3d; get_shapes3d()
).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.download import load_shapes3d_array
from models.factory import build_model, build_optimizer
from models.ewc import EWC
from representation_plasticity.stream import (
    Shapes3DDataset, FeatureOverlapLadder, FeatureConflictStream,
    FeatureIntroductionStream, FACTOR_NAMES,
)
from representation_plasticity.metrics import (
    compute_all_metrics, build_relevance_matrix, build_similarity_matrices,
    adaptation_speed, learning_curve_auc, recurrence_savings,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

RESULTS_DIR = Path(__file__).parent.parent / "results" / "representation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Parser ────────────────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Representation plasticity experiment")

    p.add_argument("--stream",     default="ladder",
                   choices=["ladder", "conflict", "introduction"])

    # Architecture
    p.add_argument("--arch",       default="mlp_2x512",
                   choices=["mlp_2x512","mlp_2x1024","mlp_3x256","mlp_4x256","mlp_4x512","cnn"])
    p.add_argument("--activation", default="relu",   choices=["relu","gelu","leaky_relu"])
    p.add_argument("--layer_norm", action="store_true")

    # Optimizer
    p.add_argument("--optimizer",  default="adam",   choices=["adam","sgd"])
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--use_ewc",    action="store_true")
    p.add_argument("--ewc_lambda", type=float, default=5000.0)

    # Stream
    p.add_argument(
        "--switch_every",
        default="2000",
        help="Steps per task. Single int (uniform) or comma-separated list of ints "
             "(one per task position), e.g. '2000' or '1000,2000,1000,...'",
    )
    p.add_argument("--n_cycles", type=int, default=2,
                   help="Number of base-cycle repetitions (FeatureOverlapLadder only)")

    # Training
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--eval_every",   type=int,   default=200)
    p.add_argument("--seed",         type=int,   default=0)

    # Shapes3D subset (use a fraction for fast iteration)
    p.add_argument("--n_samples",    type=int,   default=None,
                   help="Limit Shapes3D to n_samples (None = all 480k)")

    # Wandb
    p.add_argument("--wandb_project", default="plasticity-representation")
    p.add_argument("--wandb",   action="store_true")
    p.add_argument("--run_name", default=None)

    return p


# ── Build stream ──────────────────────────────────────────────────────────────

def _parse_switch_every(raw: str) -> int | list[int]:
    """'2000' → 2000,  '1000,2000,1000' → [1000, 2000, 1000]."""
    parts = [int(x.strip()) for x in raw.split(",")]
    return parts[0] if len(parts) == 1 else parts


def build_stream(name: str, shapes3d: Shapes3DDataset, args: argparse.Namespace):
    sw = _parse_switch_every(args.switch_every)
    if name == "ladder":
        return FeatureOverlapLadder(shapes3d, switch_every=sw, n_cycles=args.n_cycles)
    if name == "conflict":
        return FeatureConflictStream(shapes3d, switch_every=sw)
    if name == "introduction":
        return FeatureIntroductionStream(shapes3d, switch_every=sw)
    raise ValueError(f"Unknown stream: {name}")


def _make_run_name(args: argparse.Namespace) -> str:
    return (f"repr_{args.stream}_{args.arch}_{args.activation}"
            f"_ln{int(args.layer_norm)}"
            f"_opt{args.optimizer}_sw{args.switch_every}_s{args.seed}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or _make_run_name(args)

    # wandb
    wb_run = None
    if args.wandb and _WANDB_AVAILABLE:
        wb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # load shapes3d
    print("Loading Shapes3D…")
    h5 = load_shapes3d_array()
    shapes3d = Shapes3DDataset.from_h5(h5)

    if args.n_samples is not None:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(shapes3d), size=args.n_samples, replace=False)
        shapes3d = Shapes3DDataset(
            images=shapes3d.images[idx],
            latent_idx=shapes3d.latent_idx[idx],
            latent_vals=shapes3d.latent_vals[idx],
        )

    stream = build_stream(args.stream, shapes3d, args)

    # compute stream descriptors upfront
    all_tasks    = stream.tasks
    max_n_classes = max(t.n_classes for t in all_tasks)
    print(f"Computing stream descriptors for {len(all_tasks)} tasks "
          f"(max n_classes={max_n_classes})…")
    relevance_mx = build_relevance_matrix(shapes3d, all_tasks)
    feat_sim, label_sim = build_similarity_matrices(shapes3d, all_tasks)

    # build model with enough output units for the hardest task in the stream
    if args.arch == "cnn":
        model = build_model("cnn", in_channels=3, input_size=64,
                            n_classes=max_n_classes, activation=args.activation,
                            use_layer_norm=args.layer_norm)
    else:
        model = build_model(args.arch, d_in=3*64*64, n_classes=max_n_classes,
                            activation=args.activation,
                            use_layer_norm=args.layer_norm)
    model = model.to(device)

    optim = build_optimizer(model, args.optimizer, lr=args.lr)
    ewc   = EWC(model, lambda_=args.ewc_lambda) if args.use_ewc else None

    log: list[dict]         = []
    task_logs: dict[str, list] = {}
    prev_task_name: str     = ""
    curve_since_switch: list = []
    step = 0
    t0   = time.time()

    total_steps = stream.total_steps() if hasattr(stream, "total_steps") else args.switch_every * 6

    while step < total_steps:
        switched = stream.step()
        step    += 1
        task     = stream.current_task

        # rebuild dataset/loader on task switch
        if switched or step == 1:
            dataset = stream.get_dataset()
            loader  = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True
            )
            loader_iter = iter(loader)
            curve_since_switch = []
            if ewc is not None and prev_task_name:
                ewc.update(model, loader, device, n_samples=1000)

        prev_task_name = task.name

        # get batch
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        # train step — slice to current task's number of classes
        model.train()
        logits = model(x)[:, :task.n_classes]
        loss = F.cross_entropy(logits, y)
        if ewc is not None:
            loss = loss + ewc.penalty(model)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # eval
        if step % args.eval_every == 0:
            metrics = compute_all_metrics(
                model, dataset, device,
                step=step, task_name=task.name,
                extra={"train_loss": loss.item(), "elapsed": time.time() - t0},
            )
            log.append(metrics)
            curve_since_switch.append((step, metrics["current_acc"]))
            if task.name not in task_logs:
                task_logs[task.name] = []
            task_logs[task.name].append((step, metrics["current_acc"]))

            print(f"step {step:6d} | task {task.name:30s} | "
                  f"acc {metrics['current_acc']:.3f} | loss {loss.item():.4f}")
            if wb_run is not None:
                wandb.log(metrics, step=step)

    # post-hoc metrics per task
    per_task = {}
    for tname, curve in task_logs.items():
        per_task[tname] = {
            "auc":             learning_curve_auc(curve),
            "adaptation_steps": adaptation_speed(curve),
            "final_acc":       curve[-1][1] if curve else None,
        }

    results = {
        "config":        vars(args),
        "run_name":      run_name,
        "stream":        args.stream,
        "tasks":         [t.name for t in all_tasks],
        "relevance_matrix": relevance_mx.tolist(),
        "factor_names":  FACTOR_NAMES,
        "feat_sim":      feat_sim.tolist(),
        "label_sim":     label_sim.tolist(),
        "log":           log,
        "per_task":      per_task,
        "n_params":      sum(p.numel() for p in model.parameters()),
    }

    out_path = RESULTS_DIR / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    if wb_run is not None:
        wb_run.finish()

    h5.close()
    return results


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(args)
