"""
tasks/factor_rules.py — Shapes3D tasks that differ only in which factor decides a binary label.

Task t is labelled by the upper half of factor FACTORS[t % 6] (data.factory.binary_target),
cycling through the six factors in their Shapes3D order:

    task 0 floor_hue, 1 wall_hue, 2 object_hue, 3 object_size (scale),
    4 object_type (shape), 5 object_azimuth (orientation), 6 floor_hue, ...

Every task has the same images, and 2 classes, so one classification head can be
carried through the whole sequence (experiments/run_factor_rule_stream.py).

Also here: the loss over all six factors used to pretrain a representation of them,
and the distance between two tasks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.download import SHAPES3D_FACTOR_NAMES, SHAPES3D_FACTOR_N_VALUES
from data.factory import binary_target

FACTORS = list(SHAPES3D_FACTOR_NAMES)          # the cycle order = the factor order
N_FACTORS = len(FACTORS)


def task_factor(task_id: int) -> int:
    """Index of the factor whose upper half labels task `task_id`."""
    return task_id % N_FACTORS


def task_schedule(num_tasks: int) -> list[int]:
    """The relevant factor of every task 0, ..., num_tasks - 1."""
    return [task_factor(t) for t in range(num_tasks)]


def task_labels(latent_idx, factor: int):
    """Binary labels of task-rule `factor` for an (N, 6) array/tensor of latent indices."""
    return binary_target(latent_idx[:, factor], SHAPES3D_FACTOR_N_VALUES[factor])


def all_task_labels(latent_idx):
    """(N, 6): column f holds the labels of the rule of factor f."""
    return binary_target(latent_idx, SHAPES3D_FACTOR_N_VALUES)


def task_distance(prev_factor, factor) -> float:
    """
    Distance between consecutive tasks: 0 for the same relevant factor, 1 for a
    different one, NaN when there is no previous task. A placeholder for a graded
    distance between factors.
    """
    if prev_factor is None:
        return math.nan
    return 0.0 if prev_factor == factor else 1.0


class FactorCrossEntropy(nn.Module):
    """
    Sum of the cross entropies of all six Shapes3D factors, for a network whose
    output stacks one block of logits per factor (10 + 10 + 10 + 8 + 4 + 15 = 57)
    and targets that are (B, 6) latent indices:

        L = sum_f w_f * CE(logits[:, block_f], y[:, f])      (w_f = 1 by default)

    A Linear(n, 57) head over these blocks is exactly six independent heads Linear(n, K_f).
    """

    def __init__(self, n_values=SHAPES3D_FACTOR_N_VALUES, weights=None):
        super().__init__()
        self.n_values = list(n_values)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.n_values)

    @property
    def n_outputs(self) -> int:
        return sum(self.n_values)

    def split(self, logits):
        return torch.split(logits, self.n_values, dim=1)

    def per_factor(self, logits, y):
        """(loss, n_correct) of every factor in the batch."""
        return [(F.cross_entropy(lf, y[:, f]), (lf.argmax(dim=1) == y[:, f]).sum().item())
                for f, lf in enumerate(self.split(logits))]

    def forward(self, logits, y):
        return sum(w * F.cross_entropy(lf, y[:, f])
                   for f, (w, lf) in enumerate(zip(self.weights, self.split(logits))))

    def n_correct(self, logits, y):
        """Correct factor predictions averaged over the factors: accuracy = mean factor accuracy."""
        return sum((lf.argmax(dim=1) == y[:, f]).sum().item()
                   for f, lf in enumerate(self.split(logits))) / len(self.n_values)
