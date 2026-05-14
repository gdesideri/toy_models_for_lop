"""
models/ewc.py — Elastic Weight Consolidation (Kirkpatrick et al., 2017).

Usage:
    ewc = EWC(model, lambda_=5000)

    # after training on task t:
    ewc.update(model, dataloader, device)

    # during training on task t+1, add ewc penalty to the loss:
    loss = cross_entropy(logits, y) + ewc.penalty(model)
    loss.backward()
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """
    Computes and applies EWC regularization.

    After training each task, call `update()` to store the Fisher diagonal
    and the optimal parameters θ*. During the next task, add `penalty()`
    to the training loss.
    """

    def __init__(self, model: nn.Module, lambda_: float = 5000.0):
        self.lambda_ = lambda_
        # List of (fisher_dict, param_dict) tuples, one per past task.
        self._tasks: list[tuple[dict, dict]] = []

    def update(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        n_samples: Optional[int] = None,
    ) -> None:
        """
        Estimate diagonal Fisher and store θ*.

        Args:
            model:      Trained model after task t.
            dataloader: DataLoader for task-t data (x, y).
            device:     Compute device.
            n_samples:  Cap the number of samples used (None = all).
        """
        model.eval()
        fisher: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                fisher[name] = torch.zeros_like(p.data)

        seen = 0
        for x, y in dataloader:
            if n_samples is not None and seen >= n_samples:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            out = model(x)
            # use the log-probability of the predicted class as the objective
            log_prob = F.log_softmax(out, dim=1)
            # sample a label from the current predictions (online Fisher)
            sampled_y = torch.multinomial(log_prob.exp(), 1).squeeze(1)
            loss = F.nll_loss(log_prob, sampled_y)
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.data.pow(2) * x.size(0)
            seen += x.size(0)

        for name in fisher:
            fisher[name] /= max(seen, 1)

        # store θ*
        params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self._tasks.append((fisher, params))

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Return the EWC regularisation penalty for all past tasks.
        Add this to the current task's loss before calling backward().
        """
        if not self._tasks:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for fisher, params_star in self._tasks:
            for name, p in model.named_parameters():
                if name in fisher:
                    penalty = penalty + (fisher[name] * (p - params_star[name]).pow(2)).sum()

        return (self.lambda_ / 2) * penalty

    def n_tasks(self) -> int:
        return len(self._tasks)
