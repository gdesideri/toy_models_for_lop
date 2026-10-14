"""
train.py — Training loop utilities (regression, MSE loss).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tasks.toy_regression import Task, make_rng  # noqa: F401


# ── Loss ──────────────────────────────────────────────────────────────────────

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


# ── Single-task training ──────────────────────────────────────────────────────

def train_on_task(
    model: nn.Module,
    task: Task,
    n_steps: int,
    batch_size: int = 256,
    lr: float = 1e-3,
    rng_seed: int = 0,
    log_every: int = 50,
) -> dict:
    rng = make_rng(rng_seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log_loss = []

    for step in range(n_steps):
        x, y   = task.sample(batch_size, rng)
        pred   = model(x)
        loss   = mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            log_loss.append((step, loss.item()))

    return {"loss": log_loss}


# ── Gradient utilities ────────────────────────────────────────────────────────

def _flat_grad(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel()))
    return torch.cat(grads)
