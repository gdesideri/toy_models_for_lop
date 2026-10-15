"""
tasks/class_attributes.py — Binary semantic attributes of a dataset's classes, as tasks.

Each attribute splits the classes into a positive set (label 1) and the rest (label 0),
e.g. CIFAR-10 "wings" = {airplane, bird}. A task trains on a class-balanced subset: as
many positive as negative images (at most --per_label of each), drawn once per
(seed, attribute, split), so an attribute that recurs in a stream is the same task.

    task = ClassAttributeTask("cifar10", "four_legs", per_label=10_000, seed=0)
    train_task = task.apply(train_set)      # balanced subset, labels 0/1

The attributes are chosen to be visible in the image: what the object is (animal or
vehicle), its body (four legs, fur, wings) and its parts (wheels).
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Subset

from tasks.feature_streams import TaskLabels

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# dataset -> attribute -> positive class indices (the rest is label 0)
ATTRIBUTES = {
    "cifar10": {
        "animal":    (2, 3, 4, 5, 6, 7),   # bird cat deer dog frog horse   vs vehicles
        "four_legs": (3, 4, 5, 6, 7),      # cat deer dog frog horse
        "fur":       (3, 4, 5, 7),         # cat deer dog horse
        "wings":     (0, 2),               # airplane bird
        "wheels":    (1, 9),               # automobile truck
    },
}


def attribute_names(dataset_kind: str) -> list[str]:
    """The attributes defined for `dataset_kind`, in a fixed order (their factor indices)."""
    if dataset_kind not in ATTRIBUTES:
        raise ValueError(f"no class attributes for '{dataset_kind}'; defined for {list(ATTRIBUTES)}")
    return list(ATTRIBUTES[dataset_kind])


class ClassAttributeTask:
    """One attribute of `dataset_kind` as a balanced binary task."""

    def __init__(self, dataset_kind: str, name: str, per_label: int, seed: int):
        names = attribute_names(dataset_kind)
        if name not in names:
            raise ValueError(f"unknown attribute '{name}' for {dataset_kind}; available: {names}")
        self.name = name
        self.positive = ATTRIBUTES[dataset_kind][name]
        self.relevant_factors = [names.index(name)]
        self.per_label = per_label
        self.seed = seed

    def apply(self, dataset) -> TaskLabels:
        """A balanced subset of `dataset` (needs .targets) labelled by this attribute."""
        targets = np.asarray(dataset.targets)
        y = np.isin(targets, self.positive).astype(np.int64)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n = min(self.per_label, len(pos), len(neg))
        rng = np.random.default_rng([self.seed, self.relevant_factors[0], len(dataset)])
        idx = np.sort(np.concatenate([rng.choice(pos, n, replace=False),
                                      rng.choice(neg, n, replace=False)]))
        return TaskLabels(Subset(dataset, idx), y[idx], 2)
