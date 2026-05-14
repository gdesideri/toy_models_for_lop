"""
representation_plasticity/metrics.py — Learner-dependent metrics for the representation setting.

Metrics:
    current_task_accuracy       — accuracy on the current task
    adaptation_speed            — steps to reach 70% accuracy after a task switch
    learning_curve_auc          — area under the learning curve (normalised)
    recurrence_savings          — improvement in convergence when a task recurs
    conflict_cost               — accuracy drop when a conflicting task replaces a compatible one
    late_feature_acquisition    — steps to reach threshold after a new factor is introduced
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from representation_plasticity.stream import (
    Shapes3DDataset, Task, TaskDataset, interventional_relevance,
    task_similarity_feature, task_similarity_label, FACTOR_NAMES,
)


# ── Evaluation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_accuracy(
    model:   nn.Module,
    dataset: TaskDataset,
    device:  torch.device,
    batch_size: int = 512,
) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred  = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_loss(
    model:   nn.Module,
    dataset: TaskDataset,
    device:  torch.device,
    batch_size: int = 512,
) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits  = model(x)
        loss    = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        total      += y.size(0)
    return total_loss / max(total, 1)


# ── Plasticity metrics ────────────────────────────────────────────────────────

def adaptation_speed(
    accuracy_curve: list[tuple[int, float]],
    threshold: float = 0.70,
) -> Optional[int]:
    """
    Number of steps (from the task switch) to first exceed `threshold`.
    Returns None if threshold was never reached.
    """
    for step, acc in accuracy_curve:
        if acc >= threshold:
            return step
    return None


def learning_curve_auc(
    accuracy_curve: list[tuple[int, float]],
    max_steps: Optional[int] = None,
) -> float:
    """
    Normalised area under the (steps, accuracy) curve via trapezoidal integration.
    Normalised by max_steps so the result is in [0, 1].
    """
    if len(accuracy_curve) < 2:
        return float("nan")
    steps  = np.array([s for s, _ in accuracy_curve], dtype=float)
    accs   = np.array([a for _, a in accuracy_curve], dtype=float)
    total  = max_steps or steps[-1]
    return float(np.trapz(accs, steps) / total) if total > 0 else float("nan")


def recurrence_savings(
    first_curve:  list[tuple[int, float]],
    repeat_curve: list[tuple[int, float]],
    threshold: float = 0.70,
) -> Optional[float]:
    """
    Fraction of steps saved when the task is seen for the second time.
    Positive value = learner reaches threshold faster the second time.
    """
    s1 = adaptation_speed(first_curve,  threshold)
    s2 = adaptation_speed(repeat_curve, threshold)
    if s1 is None or s2 is None:
        return None
    return float((s1 - s2) / max(s1, 1))


def conflict_cost(
    acc_before_conflict: float,
    acc_after_conflict:  float,
) -> float:
    """Accuracy drop caused by the conflicting task."""
    return acc_before_conflict - acc_after_conflict


# ── Stream-level characterisation ─────────────────────────────────────────────

def build_relevance_matrix(
    shapes3d: Shapes3DDataset,
    tasks:    list[Task],
    n_samples: int = 5000,
) -> np.ndarray:
    """
    Returns (n_tasks, n_factors) matrix of interventional relevance scores.
    Row t, column j = R_{t,j}.
    """
    matrix = np.zeros((len(tasks), len(FACTOR_NAMES)))
    rng = np.random.default_rng(0)
    for i, task in enumerate(tasks):
        matrix[i] = interventional_relevance(shapes3d, task, n_samples, rng)
    return matrix


def build_similarity_matrices(
    shapes3d: Shapes3DDataset,
    tasks:    list[Task],
    n_samples: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        feat_sim  (n_tasks, n_tasks) — pairwise Jaccard on relevant factors
        label_sim (n_tasks, n_tasks) — pairwise label correlation
    """
    n = len(tasks)
    feat_sim  = np.zeros((n, n))
    label_sim = np.zeros((n, n))
    rng = np.random.default_rng(0)
    for i in range(n):
        for j in range(n):
            feat_sim[i, j]  = task_similarity_feature(tasks[i], tasks[j])
            label_sim[i, j] = task_similarity_label(shapes3d, tasks[i], tasks[j],
                                                     n_samples, rng)
    return feat_sim, label_sim


# ── Full snapshot ──────────────────────────────────────────────────────────────

def compute_all_metrics(
    model:          nn.Module,
    current_dataset: TaskDataset,
    device:         torch.device,
    step:           int,
    task_name:      str,
    extra:          Optional[dict] = None,
) -> dict:
    """Metrics logged at a single evaluation step."""
    out = {
        "step":            step,
        "task_name":       task_name,
        "current_acc":     evaluate_accuracy(model, current_dataset, device),
        "current_loss":    evaluate_loss(model, current_dataset, device),
    }
    if extra:
        out.update(extra)
    return out
