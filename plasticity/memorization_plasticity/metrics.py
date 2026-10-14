"""
memorization_plasticity/metrics.py — Learner-dependent metrics for the memorization setting.

All metrics operate on a snapshot of the model + stream state at a given
training step. They are designed to be cheap to call at regular intervals.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.randomized_labels import RandomizedLabelStream


@torch.no_grad()
def _predict_all(
    model:  nn.Module,
    stream: RandomizedLabelStream,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (predictions, current_labels) arrays for the full stream.
    """
    model.eval()
    preds, labels = [], []
    n = len(stream)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        xs   = torch.stack([stream.base[i][0] for i in range(start, end)]).to(device)
        pred = model(xs).argmax(1).cpu().numpy()
        preds.append(pred)
        labels.append(stream._current_labels[start:end])
    return np.concatenate(preds), np.concatenate(labels)


# ── Core metrics ──────────────────────────────────────────────────────────────

@torch.no_grad()
def current_label_accuracy_changed(
    model:      nn.Module,
    stream:     RandomizedLabelStream,
    device:     torch.device,
    batch_size: int = 512,
) -> float:
    """
    Accuracy on the changed subset under the CURRENT labels.
    Measures how well the model has acquired new associations.
    """
    changed = stream.get_changed_indices()
    if len(changed) == 0:
        return float("nan")
    preds, labels = _predict_all(model, stream, batch_size, device)
    return float(np.mean(preds[changed] == labels[changed]))


@torch.no_grad()
def previous_label_prediction_rate(
    model:      nn.Module,
    stream:     RandomizedLabelStream,
    device:     torch.device,
    batch_size: int = 512,
) -> float:
    """
    Rate at which the model still predicts the ORIGINAL label for changed samples.
    High value = model has not yet overwritten old associations.
    """
    changed = stream.get_changed_indices()
    if len(changed) == 0:
        return float("nan")
    preds, _ = _predict_all(model, stream, batch_size, device)
    original  = stream.get_original_labels()
    return float(np.mean(preds[changed] == original[changed]))


@torch.no_grad()
def unchanged_subset_retention(
    model:      nn.Module,
    stream:     RandomizedLabelStream,
    device:     torch.device,
    batch_size: int = 512,
) -> float:
    """
    Accuracy on the unchanged subset under the current (= original) labels.
    Measures catastrophic forgetting of stable associations.
    """
    unchanged = stream.get_unchanged_indices()
    if len(unchanged) == 0:
        return float("nan")
    preds, labels = _predict_all(model, stream, batch_size, device)
    return float(np.mean(preds[unchanged] == labels[unchanged]))


@torch.no_grad()
def overall_accuracy(
    model:      nn.Module,
    stream:     RandomizedLabelStream,
    device:     torch.device,
    batch_size: int = 512,
) -> float:
    """Overall current-label accuracy (all samples)."""
    preds, labels = _predict_all(model, stream, batch_size, device)
    return float(np.mean(preds == labels))


def overwrite_time_distribution(change_history: list[dict]) -> dict:
    """
    Analyse the history of change events to compute:
        - total changes applied
        - mean / std of labels changed per event
        - steps between events
    """
    if not change_history:
        return {}
    n_changed = [e["n_changed"] for e in change_history]
    steps     = [e["step"]      for e in change_history]
    gaps      = np.diff(steps).tolist() if len(steps) > 1 else []
    return {
        "n_events":            len(change_history),
        "total_changed":       int(sum(n_changed)),
        "mean_changed_per_event": float(np.mean(n_changed)),
        "std_changed_per_event":  float(np.std(n_changed)),
        "mean_steps_between_events": float(np.mean(gaps)) if gaps else float("nan"),
    }


# ── Full metric snapshot ──────────────────────────────────────────────────────

def compute_all_metrics(
    model:      nn.Module,
    stream:     RandomizedLabelStream,
    device:     torch.device,
    batch_size: int = 512,
) -> dict:
    """
    Returns a dictionary with all metrics at the current step.
    Suitable for logging to wandb or saving as JSON.
    """
    return {
        "step":                          stream.current_step,
        "current_label_acc_changed":     current_label_accuracy_changed(model, stream, device, batch_size),
        "previous_label_pred_rate":      previous_label_prediction_rate(model, stream, device, batch_size),
        "unchanged_retention":           unchanged_subset_retention(model, stream, device, batch_size),
        "overall_accuracy":              overall_accuracy(model, stream, device, batch_size),
        "n_change_events":               len(stream.get_change_history()),
        "mean_persistence":              float(np.mean(stream.association_persistence())),
    }
