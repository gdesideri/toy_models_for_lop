"""
probe.py — LoP measurement via fresh-model gap (regression, MSE).

Five fixed probe tasks use the full input space (no suppressed dims) with a
random unit-norm target w.  LoP = MSE(prefix) - MSE(fresh) after n_steps.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from src.benchmark import Task, make_rng
from src.train import train_on_task, mse_loss


def build_probe_bank(k: int = 8) -> dict[str, Task]:
    """Five full-space probe tasks (suppressed_dims=[]), fixed seeds."""
    probes = {}
    for name, seed in [("probe_0", 1000), ("probe_1", 1001), ("probe_2", 1002),
                       ("probe_3", 1003), ("probe_4", 1004)]:
        rng = make_rng(seed)
        w = torch.randn(k, generator=rng)
        w = w / w.norm()
        probes[name] = Task(suppressed_dims=[], w=w, noise_std=0.02)
    return probes


def fresh_model_copy(model: nn.Module) -> nn.Module:
    fresh = copy.deepcopy(model)
    for layer in fresh.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return fresh


def evaluate_probe(
    model: nn.Module,
    probe_task: Task,
    n_steps: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    rng_seed: int = 42,
) -> dict:
    prefix_model = copy.deepcopy(model)
    prefix_log   = train_on_task(prefix_model, probe_task, n_steps=n_steps,
                                 batch_size=batch_size, lr=lr, rng_seed=rng_seed)
    fresh        = fresh_model_copy(model)
    fresh_log    = train_on_task(fresh, probe_task, n_steps=n_steps,
                                 batch_size=batch_size, lr=lr, rng_seed=rng_seed)

    pc = prefix_log["loss"]
    fc = fresh_log["loss"]
    final_gap = (pc[-1][1] - fc[-1][1]) if pc and fc else float("nan")
    return {"prefix_curve": pc, "fresh_curve": fc, "final_gap": final_gap}


def run_probe_bank(model, probe_bank, n_steps=400, batch_size=256,
                   lr=1e-3, rng_seed=42):
    return {name: evaluate_probe(model, task, n_steps=n_steps,
                                 batch_size=batch_size, lr=lr, rng_seed=rng_seed)
            for name, task in probe_bank.items()}
