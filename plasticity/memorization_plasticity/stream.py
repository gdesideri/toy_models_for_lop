"""
memorization_plasticity/stream.py — Non-stationary label stream for memorization plasticity.

The input distribution is fixed (CIFAR-10 or MNIST); only the label assignment
changes over time according to four parameters:

    p_change   : fraction of the mutable support whose labels are randomised
                 at each change event (target drift magnitude).
    L          : number of training steps between consecutive change events
                 (association change rate = p_change / L).
    H_random   : relabeling entropy — number of classes the new label can be
                 drawn from uniformly (1 = deterministic flip, n_classes = full).
    p_mutable  : fraction of the dataset that is eligible to have its label
                 changed (affected support size).

Stream descriptors (computed from the parameters, independent of any learner):
    - target_drift_magnitude  = p_change
    - association_change_rate = p_change / L
    - relabeling_entropy      = log2(H_random)
    - affected_support_size   = p_mutable
    - association_persistence : per-sample, how many steps until label changes
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class StreamConfig:
    p_change:  float = 0.1    # fraction of mutable support randomised per event
    L:         int   = 500    # steps between change events
    H_random:  int   = 10     # relabeling entropy (number of target classes)
    p_mutable: float = 1.0    # fraction of dataset eligible for label change
    n_classes: int   = 10
    seed:      int   = 42

    @property
    def association_change_rate(self) -> float:
        return self.p_change / self.L

    @property
    def relabeling_entropy(self) -> float:
        return math.log2(max(self.H_random, 1))

    def describe(self) -> dict:
        return {
            "p_change":               self.p_change,
            "L":                      self.L,
            "H_random":               self.H_random,
            "p_mutable":              self.p_mutable,
            "association_change_rate": self.association_change_rate,
            "relabeling_entropy":     self.relabeling_entropy,
        }


class RandomizedLabelStream:
    """
    Wraps an existing classification dataset and maintains a dynamic label
    mapping that changes periodically.

    Usage:
        stream = RandomizedLabelStream(base_dataset, cfg)
        stream.step()                       # advance one step; may relabel
        x, y_current = stream[i]            # get sample with CURRENT label
        stream.get_original_labels()        # original dataset labels
        stream.get_current_labels()         # current (possibly relabelled) labels
        stream.get_change_history()         # list of change events
    """

    def __init__(self, base_dataset: Dataset, cfg: StreamConfig):
        self.base  = base_dataset
        self.cfg   = cfg
        self.rng   = np.random.default_rng(cfg.seed)

        n = len(base_dataset)  # type: ignore[arg-type]
        self._original_labels = self._extract_labels(base_dataset)
        self._current_labels  = self._original_labels.copy()

        # mark which indices are mutable
        n_mutable = max(1, int(n * cfg.p_mutable))
        all_idx   = np.arange(n)
        self.rng.shuffle(all_idx)
        self._mutable_mask = np.zeros(n, dtype=bool)
        self._mutable_mask[all_idx[:n_mutable]] = True

        # tracking
        self._step:    int  = 0
        self._changes: list[dict] = []   # history of change events
        self._last_changed_at = np.full(n, -1, dtype=np.int64)  # step of last change

    # ── Label utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_labels(dataset: Dataset) -> np.ndarray:
        if hasattr(dataset, "targets"):
            t = dataset.targets
            return np.array(t.numpy() if isinstance(t, torch.Tensor) else t, dtype=np.int64)
        labels = []
        for _, y in dataset:  # type: ignore[var-annotated]
            labels.append(int(y))
        return np.array(labels, dtype=np.int64)

    # ── Stream advance ────────────────────────────────────────────────────────

    def step(self) -> bool:
        """
        Advance the stream by one training step.

        Returns True if a label-change event occurred this step.
        """
        self._step += 1
        if self._step % self.cfg.L != 0:
            return False

        mutable_idx = np.where(self._mutable_mask)[0]
        n_change    = max(1, int(len(mutable_idx) * self.cfg.p_change))
        chosen      = self.rng.choice(mutable_idx, size=n_change, replace=False)

        old_labels = self._current_labels[chosen].copy()
        new_labels = self._sample_new_labels(old_labels)
        self._current_labels[chosen]  = new_labels
        self._last_changed_at[chosen] = self._step

        self._changes.append({
            "step":          self._step,
            "n_changed":     n_change,
            "indices":       chosen.tolist(),
            "old_labels":    old_labels.tolist(),
            "new_labels":    new_labels.tolist(),
        })
        return True

    def _sample_new_labels(self, old_labels: np.ndarray) -> np.ndarray:
        n  = self.cfg.n_classes
        h  = min(self.cfg.H_random, n)
        new = np.empty_like(old_labels)
        for i, old in enumerate(old_labels):
            # choose from H_random classes excluding the current one
            candidates = np.array([c for c in range(n) if c != old], dtype=np.int64)
            if h < n:
                candidates = candidates[: h - 1]   # keep entropy bounded
            new[i] = self.rng.choice(candidates)
        return new

    # ── Access ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._original_labels)

    def __getitem__(self, idx: int) -> tuple:
        x, _ = self.base[idx]
        y    = int(self._current_labels[idx])
        return x, y

    def get_current_labels(self) -> np.ndarray:
        return self._current_labels.copy()

    def get_original_labels(self) -> np.ndarray:
        return self._original_labels.copy()

    def get_changed_mask(self) -> np.ndarray:
        """True for samples whose current label differs from original."""
        return self._current_labels != self._original_labels

    def get_changed_indices(self) -> np.ndarray:
        return np.where(self.get_changed_mask())[0]

    def get_unchanged_indices(self) -> np.ndarray:
        return np.where(~self.get_changed_mask())[0]

    def get_change_history(self) -> list[dict]:
        return list(self._changes)

    def association_persistence(self) -> np.ndarray:
        """
        For each sample, number of steps the current label has been valid.
        Samples that have never changed report persistence = current_step + 1.
        """
        persistence = np.where(
            self._last_changed_at >= 0,
            self._step - self._last_changed_at,
            self._step + 1,
        )
        return persistence

    @property
    def current_step(self) -> int:
        return self._step

    def as_dataloader(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
