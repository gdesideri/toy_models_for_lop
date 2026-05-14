"""
memorization_plasticity/run.py — Experiment runner for the memorization setting.

Run:
    uv run python -m memorization_plasticity.run --dataset cifar10 --arch mlp_2x512 \
        --activation relu --optimizer adam --p_change 0.1 --L 500
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.download import get_mnist, get_cifar10
from models.factory import build_model, build_optimizer
from models.ewc import EWC
from memorization_plasticity.stream import RandomizedLabelStream, StreamConfig
from memorization_plasticity.metrics import compute_all_metrics

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

RESULTS_DIR = Path(__file__).parent.parent / "results" / "memorization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Config ────────────────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memorization plasticity experiment")

    # Dataset
    p.add_argument("--dataset",   default="cifar10", choices=["cifar10", "mnist"])

    # Architecture
    p.add_argument("--arch",      default="mlp_2x512",
                   choices=["mlp_2x512","mlp_2x1024","mlp_3x256","mlp_4x256","mlp_4x512","cnn"])
    p.add_argument("--activation", default="relu",   choices=["relu","gelu","leaky_relu"])
    p.add_argument("--layer_norm", action="store_true")

    # Optimizer
    p.add_argument("--optimizer",  default="adam",   choices=["adam","sgd"])
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--use_ewc",    action="store_true")
    p.add_argument("--ewc_lambda", type=float, default=5000.0)

    # Stream parameters
    p.add_argument("--p_change",   type=float, default=0.1)
    p.add_argument("--L",          type=int,   default=500)
    p.add_argument("--H_random",   type=int,   default=10)
    p.add_argument("--p_mutable",  type=float, default=1.0)

    # Training
    p.add_argument("--n_steps",    type=int,   default=20000)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--eval_every", type=int,   default=200)
    p.add_argument("--seed",       type=int,   default=0)

    # Wandb
    p.add_argument("--wandb_project", default="plasticity-memorization")
    p.add_argument("--wandb",  action="store_true")
    p.add_argument("--run_name", default=None)

    return p


# ── Training utilities ────────────────────────────────────────────────────────

def _make_run_name(args: argparse.Namespace) -> str:
    return (f"{args.dataset}_{args.arch}_{args.activation}"
            f"_ln{int(args.layer_norm)}"
            f"_opt{args.optimizer}_lr{args.lr}"
            f"_pch{args.p_change}_L{args.L}_H{args.H_random}_pm{args.p_mutable}"
            f"_s{args.seed}")


def _build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.dataset == "cifar10":
        d_in, in_ch, sz = 3072, 3, 32
    else:
        d_in, in_ch, sz = 784, 1, 28

    if args.arch == "cnn":
        model = build_model("cnn", in_channels=in_ch, input_size=sz,
                            n_classes=10, activation=args.activation,
                            use_layer_norm=args.layer_norm)
    else:
        model = build_model(args.arch, d_in=d_in, n_classes=10,
                            activation=args.activation,
                            use_layer_norm=args.layer_norm)
    return model.to(device)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
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

    # data
    if args.dataset == "cifar10":
        train_ds, _ = get_cifar10()
    else:
        train_ds, _ = get_mnist(flatten=False)

    cfg    = StreamConfig(
        p_change=args.p_change, L=args.L,
        H_random=args.H_random, p_mutable=args.p_mutable,
        n_classes=10, seed=args.seed,
    )
    stream = RandomizedLabelStream(train_ds, cfg)
    loader = stream.as_dataloader(args.batch_size, shuffle=True)
    loader_iter = iter(loader)

    # model + optimizer
    model  = _build_model(args, device)
    optim  = build_optimizer(model, args.optimizer, lr=args.lr)
    ewc    = EWC(model, lambda_=args.ewc_lambda) if args.use_ewc else None

    log: list[dict] = []
    t0 = time.time()

    for step in range(1, args.n_steps + 1):
        # advance stream (may relabel)
        changed = stream.step()

        # get batch
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        # forward + loss
        model.train()
        logits = model(x)
        loss   = F.cross_entropy(logits, y)
        if ewc is not None:
            loss = loss + ewc.penalty(model)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # if EWC and a change event just fired, update Fisher
        if ewc is not None and changed:
            ewc.update(model, loader, device, n_samples=2000)

        # eval
        if step % args.eval_every == 0:
            metrics = compute_all_metrics(model, stream, device)
            metrics["train_loss"] = loss.item()
            metrics["elapsed"]    = time.time() - t0
            log.append(metrics)
            print(f"step {step:6d} | loss {loss.item():.4f} | "
                  f"acc_changed {metrics['current_label_acc_changed']:.3f} | "
                  f"retention {metrics['unchanged_retention']:.3f}")
            if wb_run is not None:
                wandb.log(metrics, step=step)

    results = {
        "config":     vars(args),
        "stream":     cfg.describe(),
        "run_name":   run_name,
        "log":        log,
        "n_params":   sum(p.numel() for p in model.parameters()),
    }

    out_path = RESULTS_DIR / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    if wb_run is not None:
        wb_run.finish()

    return results


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(args)
