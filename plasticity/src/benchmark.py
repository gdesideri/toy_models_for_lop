"""
benchmark.py — Data generation for the disjoint subspace plasticity experiment.

Setup:
  k = 8 latent/input dimensions, rank = 2 active dims per task.
  4 tasks with mutually exclusive active subspaces: {0,1}, {2,3}, {4,5}, {6,7}.
  A = I  (no mixing matrix): x is sampled directly in R^k.
  Target: w = ones(k)/sqrt(k)  — uniform, so every task is equally informative.
  y = sign(w · x)  — binary classification label.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


# ── RNG ──────────────────────────────────────────────────────────────────────

def make_rng(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── Sampler ───────────────────────────────────────────────────────────────────

def sample_coverage_hole(n: int, k: int, rng: torch.Generator,
                         suppressed_dims: list[int] | None = None) -> torch.Tensor:
    """z ~ N(0, I_k) with suppressed_dims zeroed out."""
    if suppressed_dims is None:
        suppressed_dims = list(range(k // 2, k))
    z = torch.randn(n, k, generator=rng)
    for d in suppressed_dims:
        z[:, d] = 0.0
    return z


# ── Task ──────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    """
    One task in the disjoint stream.
    x = z + noise,  z ~ N(0,I) with suppressed_dims = 0.
    y = sign(w · x).
    """
    suppressed_dims: list[int] = field(default_factory=list)
    w: torch.Tensor = field(default_factory=lambda: torch.zeros(8))
    noise_std: float = 0.02

    def sample(self, n: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.w.shape[0]
        z = sample_coverage_hole(n, k, rng, self.suppressed_dims)
        noise = torch.randn(n, k, generator=rng) * self.noise_std
        x = z + noise
        y = (x @ self.w) ** 2   # nonlinear target: squared projection onto w
        return x, y


# ── Task factory ──────────────────────────────────────────────────────────────

def make_disjoint_direct_tasks(
    k: int = 8,
    rank: int = 2,
    noise_std: float = 0.02,
    shared_w: Optional[torch.Tensor] = None,
) -> list[Task]:
    """
    Creates k//rank tasks with mutually exclusive active subspaces.

    Task i has non-zero variance ONLY in dims [i*rank : (i+1)*rank].
    All tasks share the same target w (default: uniform 1/sqrt(k)).

    Because A = I, the fc1 weight vectors operate directly on input dims —
    subspace alignment is measurable without any projection.
    """
    assert k % rank == 0, f"k ({k}) must be divisible by rank ({rank})"
    n_tasks = k // rank

    if shared_w is None:
        shared_w = torch.ones(k) / math.sqrt(k)  # unit-norm, equal weight per dim

    tasks = []
    for i in range(n_tasks):
        active     = list(range(i * rank, (i + 1) * rank))
        suppressed = [d for d in range(k) if d not in active]
        tasks.append(Task(
            suppressed_dims=suppressed,
            w=shared_w.clone(),
            noise_std=noise_std,
        ))
    return tasks
