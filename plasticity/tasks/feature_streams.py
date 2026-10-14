"""
tasks/feature_streams.py — Non-stationary feature streams on Shapes3D.

Each task T_t is defined by:
    - R_t  : subset of latent factors that are predictive
    - g_t  : labeling rule  y = g_t(z_{R_t})

Three stream types:
    FeatureOverlapLadder       — consecutive tasks share partial feature sets
    FeatureConflictStream      — same factor appears with incompatible roles
    FeatureIntroductionStream  — new factors become predictive only later

Parity tasks, for experiments/run_tasks_stream.py:
    FeatureParityTask          — label = parity of the summed value indices of a feature set
    TwoFeatureParityTask       — the same over exactly two features
    FeatureSumTask             — "sum above median": class 1 if the summed scaled feature values
                                 are in the upper half (notes/sum_above_median_labels.md)

Shapes3D latent factors (index → name):
    0  floor_hue      (10 values, uniform in [0,1])
    1  wall_hue       (10 values)
    2  object_hue     (10 values)
    3  object_size    (8 values)
    4  object_type    (4 values: cube/cylinder/sphere/capsule)
    5  object_azimuth (15 values)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


FACTOR_NAMES = [
    "floor_hue",
    "wall_hue",
    "object_hue",
    "object_size",
    "object_type",
    "object_azimuth",
]
FACTOR_N_VALUES = [10, 10, 10, 8, 4, 15]
N_FACTORS = len(FACTOR_NAMES)


# ── Task definition ───────────────────────────────────────────────────────────

@dataclass
class Task:
    """
    A single task defined by relevant factors and a labeling rule.

    Args:
        relevant_factors: indices of latent factors that determine y.
        label_fn:         callable (latent_values_array) → int label.
        name:             human-readable task name.
    """
    relevant_factors: list[int]
    label_fn: Callable[[np.ndarray], int]
    name: str = ""
    n_classes: int = 2


@dataclass
class Shapes3DDataset:
    """
    In-memory Shapes3D dataset with pre-loaded images and latent labels.

    Shapes3D has 480 000 images indexed by a 6-tuple of latent values.
    The canonical ordering is:
        index = floor_hue * (10*10*8*4*15) + wall_hue * (10*8*4*15) + ...

    We store latent indices (integer) rather than continuous values.
    """
    images:       np.ndarray   # (N, 64, 64, 3)  uint8
    latent_idx:   np.ndarray   # (N, 6)           int
    latent_vals:  np.ndarray   # (N, 6)           float32 (normalised)

    @classmethod
    def from_h5(cls, h5_file) -> "Shapes3DDataset":
        """Load from an open h5py file handle returned by data.download.load_shapes3d_array()."""
        images      = h5_file["images"][:]           # (480000, 64, 64, 3)
        latent_vals = h5_file["labels"][:]           # (480000, 6)  float32

        # convert continuous values back to integer indices
        # using the known number of discrete values per factor
        latent_idx = np.zeros_like(latent_vals, dtype=np.int64)
        for j, n in enumerate(FACTOR_N_VALUES):
            v_min = latent_vals[:, j].min()
            v_max = latent_vals[:, j].max()
            if v_max > v_min:
                latent_idx[:, j] = np.round(
                    (latent_vals[:, j] - v_min) / (v_max - v_min) * (n - 1)
                ).astype(np.int64)

        return cls(images=images, latent_idx=latent_idx, latent_vals=latent_vals)

    def __len__(self) -> int:
        return len(self.images)

    def apply_task(self, task: Task) -> np.ndarray:
        """Return integer labels for all images under the given task."""
        return np.array([task.label_fn(self.latent_idx[i]) for i in range(len(self))],
                        dtype=np.int64)


class TaskDataset(Dataset):
    """PyTorch Dataset for a specific Shapes3D task."""

    def __init__(
        self,
        shapes3d: Shapes3DDataset,
        task:     Task,
        transform=None,
        indices:  Optional[np.ndarray] = None,
    ):
        self.shapes3d  = shapes3d
        self.task      = task
        self.transform = transform
        self.indices   = indices if indices is not None else np.arange(len(shapes3d))
        self.labels    = shapes3d.apply_task(task)[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img = self.shapes3d.images[idx]              # (64, 64, 3) uint8
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


# ── Stream descriptors ────────────────────────────────────────────────────────

def interventional_relevance(
    shapes3d: Shapes3DDataset,
    task:     Task,
    n_samples: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute R_{t,j} = Pr[g_t(z) ≠ g_t(z_{-j}, z'_j)] for each factor j.

    Estimated by Monte-Carlo: sample z, sample z'_j independently,
    compare g_t(z) and g_t(z with j replaced by z'_j).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n    = len(shapes3d)
    idx  = rng.integers(0, n, size=n_samples)
    z    = shapes3d.latent_idx[idx]       # (n_samples, 6)
    y    = np.array([task.label_fn(z[i]) for i in range(n_samples)])

    scores = np.zeros(N_FACTORS)
    for j in range(N_FACTORS):
        # replace factor j with a random other value
        z_prime      = z.copy()
        z_prime[:, j] = rng.integers(0, FACTOR_N_VALUES[j], size=n_samples)
        y_prime      = np.array([task.label_fn(z_prime[i]) for i in range(n_samples)])
        scores[j]    = float(np.mean(y != y_prime))

    return scores


def task_similarity_feature(task_a: Task, task_b: Task) -> float:
    """Jaccard overlap of relevant factor sets."""
    a, b = set(task_a.relevant_factors), set(task_b.relevant_factors)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def task_similarity_label(
    shapes3d:  Shapes3DDataset,
    task_a:    Task,
    task_b:    Task,
    n_samples: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Pearson correlation between the two tasks' label vectors."""
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, len(shapes3d), size=n_samples)
    z   = shapes3d.latent_idx[idx]
    ya  = np.array([task_a.label_fn(z[i]) for i in range(n_samples)], dtype=float)
    yb  = np.array([task_b.label_fn(z[i]) for i in range(n_samples)], dtype=float)
    if ya.std() == 0 or yb.std() == 0:
        return 0.0
    return float(np.corrcoef(ya, yb)[0, 1])


# ── Factor name aliases (for readability in task definitions) ─────────────────

SHAPE       = 4   # object_type    (4 values)
OBJ_COLOR   = 2   # object_hue     (10 values)
SCALE       = 3   # object_size    (8 values)
ORIENTATION = 5   # object_azimuth (15 values)
FLOOR_COLOR = 0   # floor_hue      (10 values)
WALL_COLOR  = 1   # wall_hue       (10 values)


# ── Built-in labeling rules ───────────────────────────────────────────────────

def _bin(latent_idx: np.ndarray, factor: int) -> int:
    """Binary split at the median index of the given factor."""
    return int(latent_idx[factor] >= FACTOR_N_VALUES[factor] // 2)


def _xor(latent_idx: np.ndarray, f1: int, f2: int) -> int:
    return int(_bin(latent_idx, f1) ^ _bin(latent_idx, f2))


def _bits2(latent_idx: np.ndarray, f1: int, f2: int) -> int:
    """2-bit label: encodes both factors → 4 classes."""
    return _bin(latent_idx, f1) * 2 + _bin(latent_idx, f2)


def _bits3(latent_idx: np.ndarray, f1: int, f2: int, f3: int) -> int:
    """3-bit label: encodes three factors → 8 classes."""
    return _bin(latent_idx, f1) * 4 + _bin(latent_idx, f2) * 2 + _bin(latent_idx, f3)


# ── Per-task step length helper ───────────────────────────────────────────────

def _resolve_step_counts(switch_every: int | list[int], n_tasks: int) -> list[int]:
    """
    Returns a list of per-task step counts.
    - int  → same length for every task
    - list → used as-is; must have length == n_tasks
    """
    if isinstance(switch_every, int):
        return [switch_every] * n_tasks
    if len(switch_every) != n_tasks:
        raise ValueError(
            f"switch_every list has {len(switch_every)} entries but stream has {n_tasks} tasks."
        )
    return list(switch_every)


def _task_idx_at_step(step: int, boundaries: list[int]) -> int:
    """0-indexed task for the given step (1-based). Clamps to last task."""
    for i, b in enumerate(boundaries):
        if step <= b:
            return i
    return len(boundaries) - 1


# ── Three stream types ────────────────────────────────────────────────────────

class FeatureOverlapLadder:
    """
    Consecutive tasks share exactly one relevant factor (Jaccard = 1/3).
    The 6-task base cycle repeats n_cycles times (default 2 → 12 task presentations).

    Base cycle (T1–T6), then repeats:
        T1 : {shape, obj_color}        label = binarise(shape)
        T2 : {obj_color, scale}        label = binarise(obj_color)
        T3 : {scale, orientation}      label = binarise(scale)
        T4 : {orientation, floor_color} label = binarise(orientation)
        T5 : {floor_color, wall_color} label = binarise(floor_color)
        T6 : {wall_color, shape}       label = binarise(wall_color)

    Args:
        switch_every: steps per task — int (uniform) or list[int] of length
                      6 * n_cycles for per-presentation control.
        n_cycles:     how many times the 6-task base cycle repeats (default 2).
    """

    _BASE: list[tuple[list[int], int, str]] = [
        ([SHAPE,       OBJ_COLOR],   SHAPE,       "shape+obj_color"),
        ([OBJ_COLOR,   SCALE],       OBJ_COLOR,   "obj_color+scale"),
        ([SCALE,       ORIENTATION], SCALE,       "scale+orientation"),
        ([ORIENTATION, FLOOR_COLOR], ORIENTATION, "orientation+floor_color"),
        ([FLOOR_COLOR, WALL_COLOR],  FLOOR_COLOR, "floor_color+wall_color"),
        ([WALL_COLOR,  SHAPE],       WALL_COLOR,  "wall_color+shape"),
    ]

    def __init__(
        self,
        shapes3d:     Shapes3DDataset,
        switch_every: int | list[int] = 2000,
        n_cycles:     int = 2,
        transform=None,
    ):
        self.shapes3d  = shapes3d
        self.n_cycles  = n_cycles
        self.transform = transform

        base_tasks = [
            Task(
                relevant_factors=factors,
                label_fn=(lambda lat, f=label_f: _bin(lat, f)),
                name=f"ladder_{name}",
                n_classes=2,
            )
            for factors, label_f, name in self._BASE
        ]
        self.tasks: list[Task] = base_tasks * n_cycles

        self._step_counts = _resolve_step_counts(switch_every, len(self.tasks))
        self._boundaries  = []
        cumsum = 0
        for s in self._step_counts:
            cumsum += s
            self._boundaries.append(cumsum)

        self._step    = 0
        self._task_id = 0

    @property
    def current_task(self) -> Task:
        return self.tasks[self._task_id]

    def step(self) -> bool:
        self._step += 1
        new_id   = _task_idx_at_step(self._step, self._boundaries)
        switched = new_id != self._task_id
        self._task_id = new_id
        return switched

    def get_dataset(self) -> TaskDataset:
        return TaskDataset(self.shapes3d, self.current_task, self.transform)

    def total_steps(self) -> int:
        return sum(self._step_counts)


class FeatureConflictStream:
    """
    The same factors reappear with incompatible or inverted predictive roles.

    12-task sequence:
        T1 : shape                         label = bin(shape)
        T2 : obj_color                     label = bin(obj_color)
        T3 : shape XOR obj_color           label = xor(shape, obj_color)
        T4 : NOT shape                     label = 1 - bin(shape)
        T5 : NOT obj_color                 label = 1 - bin(obj_color)
        T6 : floor_color                   label = bin(floor_color)
        T7 : shape XOR floor_color         label = xor(shape, floor_color)
        T8 : NOT floor_color               label = 1 - bin(floor_color)
        T9 : orientation                   label = bin(orientation)
        T10: obj_color XOR orientation     label = xor(obj_color, orientation)
        T11: NOT orientation               label = 1 - bin(orientation)
        T12: shape                         label = bin(shape)   (returns to T1)

    Args:
        switch_every: steps per task — int (uniform) or list[int] of length 12.
    """

    def __init__(
        self,
        shapes3d:     Shapes3DDataset,
        switch_every: int | list[int] = 2000,
        transform=None,
    ):
        self.shapes3d  = shapes3d
        self.transform = transform

        self.tasks: list[Task] = [
            Task([SHAPE],                lambda lat: _bin(lat, SHAPE),                        "conflict_T01_shape",             2),
            Task([OBJ_COLOR],            lambda lat: _bin(lat, OBJ_COLOR),                   "conflict_T02_obj_color",         2),
            Task([SHAPE, OBJ_COLOR],     lambda lat: _xor(lat, SHAPE, OBJ_COLOR),            "conflict_T03_shape_xor_color",   2),
            Task([SHAPE],                lambda lat: 1 - _bin(lat, SHAPE),                   "conflict_T04_not_shape",         2),
            Task([OBJ_COLOR],            lambda lat: 1 - _bin(lat, OBJ_COLOR),               "conflict_T05_not_color",         2),
            Task([FLOOR_COLOR],          lambda lat: _bin(lat, FLOOR_COLOR),                 "conflict_T06_floor_color",       2),
            Task([SHAPE, FLOOR_COLOR],   lambda lat: _xor(lat, SHAPE, FLOOR_COLOR),          "conflict_T07_shape_xor_floor",   2),
            Task([FLOOR_COLOR],          lambda lat: 1 - _bin(lat, FLOOR_COLOR),             "conflict_T08_not_floor",         2),
            Task([ORIENTATION],          lambda lat: _bin(lat, ORIENTATION),                 "conflict_T09_orientation",       2),
            Task([OBJ_COLOR, ORIENTATION], lambda lat: _xor(lat, OBJ_COLOR, ORIENTATION),   "conflict_T10_color_xor_orient",  2),
            Task([ORIENTATION],          lambda lat: 1 - _bin(lat, ORIENTATION),             "conflict_T11_not_orient",        2),
            Task([SHAPE],                lambda lat: _bin(lat, SHAPE),                        "conflict_T12_shape",             2),
        ]

        self._step_counts = _resolve_step_counts(switch_every, len(self.tasks))
        self._boundaries  = []
        cumsum = 0
        for s in self._step_counts:
            cumsum += s
            self._boundaries.append(cumsum)

        self._step    = 0
        self._task_id = 0

    @property
    def current_task(self) -> Task:
        return self.tasks[self._task_id]

    def step(self) -> bool:
        self._step += 1
        new_id   = _task_idx_at_step(self._step, self._boundaries)
        switched = new_id != self._task_id
        self._task_id = new_id
        return switched

    def get_dataset(self) -> TaskDataset:
        return TaskDataset(self.shapes3d, self.current_task, self.transform)

    def total_steps(self) -> int:
        return sum(self._step_counts)


class FeatureIntroductionStream:
    """
    New latent factors are introduced progressively; factors are later combined,
    dropped, and re-introduced in new combinations.

    12-task sequence (T1 = shape only; T2–T12 add combinations):
        T1 : shape                           2 classes
        T2 : shape + obj_color               4 classes
        T3 : shape + obj_color + scale       8 classes
        T4 : obj_color + scale               4 classes
        T5 : scale + orientation             4 classes
        T6 : scale + orientation + floor     8 classes
        T7 : orientation + floor_color       4 classes
        T8 : floor_color + wall_color        4 classes
        T9 : floor_color + wall_color + shape 8 classes
        T10: wall_color + shape              4 classes
        T11: shape + scale                   4 classes
        T12: obj_color + orientation         4 classes

    Label encoding:
        1 factor  → binary (2 classes)
        2 factors → 2-bit  (4 classes): bin(f1)*2 + bin(f2)
        3 factors → 3-bit  (8 classes): bin(f1)*4 + bin(f2)*2 + bin(f3)

    Args:
        switch_every: steps per task — int (uniform) or list[int] of length 12.
    """

    def __init__(
        self,
        shapes3d:     Shapes3DDataset,
        switch_every: int | list[int] = 2000,
        transform=None,
    ):
        self.shapes3d  = shapes3d
        self.transform = transform

        self.tasks: list[Task] = [
            Task([SHAPE],                          lambda lat: _bin(lat, SHAPE),                                    "intro_T01_shape",                    2),
            Task([SHAPE, OBJ_COLOR],               lambda lat: _bits2(lat, SHAPE, OBJ_COLOR),                      "intro_T02_shape+color",              4),
            Task([SHAPE, OBJ_COLOR, SCALE],        lambda lat: _bits3(lat, SHAPE, OBJ_COLOR, SCALE),               "intro_T03_shape+color+scale",        8),
            Task([OBJ_COLOR, SCALE],               lambda lat: _bits2(lat, OBJ_COLOR, SCALE),                      "intro_T04_color+scale",              4),
            Task([SCALE, ORIENTATION],             lambda lat: _bits2(lat, SCALE, ORIENTATION),                    "intro_T05_scale+orient",             4),
            Task([SCALE, ORIENTATION, FLOOR_COLOR],lambda lat: _bits3(lat, SCALE, ORIENTATION, FLOOR_COLOR),       "intro_T06_scale+orient+floor",       8),
            Task([ORIENTATION, FLOOR_COLOR],       lambda lat: _bits2(lat, ORIENTATION, FLOOR_COLOR),              "intro_T07_orient+floor",             4),
            Task([FLOOR_COLOR, WALL_COLOR],        lambda lat: _bits2(lat, FLOOR_COLOR, WALL_COLOR),               "intro_T08_floor+wall",               4),
            Task([FLOOR_COLOR, WALL_COLOR, SHAPE], lambda lat: _bits3(lat, FLOOR_COLOR, WALL_COLOR, SHAPE),        "intro_T09_floor+wall+shape",         8),
            Task([WALL_COLOR, SHAPE],              lambda lat: _bits2(lat, WALL_COLOR, SHAPE),                     "intro_T10_wall+shape",               4),
            Task([SHAPE, SCALE],                   lambda lat: _bits2(lat, SHAPE, SCALE),                          "intro_T11_shape+scale",              4),
            Task([OBJ_COLOR, ORIENTATION],         lambda lat: _bits2(lat, OBJ_COLOR, ORIENTATION),                "intro_T12_color+orient",             4),
        ]

        self._step_counts = _resolve_step_counts(switch_every, len(self.tasks))
        self._boundaries  = []
        cumsum = 0
        for s in self._step_counts:
            cumsum += s
            self._boundaries.append(cumsum)

        self._step    = 0
        self._task_id = 0

    @property
    def current_task(self) -> Task:
        return self.tasks[self._task_id]

    def step(self) -> bool:
        self._step += 1
        new_id   = _task_idx_at_step(self._step, self._boundaries)
        switched = new_id != self._task_id
        self._task_id = new_id
        return switched

    def get_dataset(self) -> TaskDataset:
        return TaskDataset(self.shapes3d, self.current_task, self.transform)

    def total_steps(self) -> int:
        return sum(self._step_counts)


# ── Parity tasks over feature sets (experiments/run_tasks_stream.py) ──────────

class FeatureParityTask(Task):
    """
    Binary task labelled by the parity of a set of features: label 1 if the value
    indices of `factors` sum to an even number, 0 if odd.

    With one factor this is the per-feature parity rule of the feature-switch
    streams (even index -> 1). With several, the label depends on all of them (it
    is the XNOR of their parities) and stays balanced as long as one of them has
    an even number of values -- every Shapes3D factor but azimuth (15) does.

    Works on the latent indices of data.download.Shapes3DImageDataset, or of any dataset
    with a latent_idx array (data.pca_factors.PCAFactorDataset):
        task = FeatureParityTask([4])            # object-type parity
        train_task = task.apply(train_set)       # same images, this task's labels
    """

    def __init__(self, factors):
        factors = [int(f) for f in factors]
        if not factors or len(set(factors)) != len(factors) or min(factors) < 0:
            raise ValueError(f"factors must be distinct non-negative factor indices, got {factors}")
        names = [FACTOR_NAMES[f] if f < N_FACTORS else str(f) for f in factors]
        super().__init__(relevant_factors=factors, label_fn=self._label,
                         name="parity(" + "+".join(names) + ")", n_classes=2)

    def _label(self, latent_idx: np.ndarray) -> int:
        return int(sum(int(latent_idx[f]) for f in self.relevant_factors) % 2 == 0)

    def labels(self, latent_idx: np.ndarray) -> np.ndarray:
        """Vectorised labels for an (N, F) array of latent indices."""
        return (np.asarray(latent_idx)[:, self.relevant_factors].sum(axis=1) % 2 == 0).astype(np.int64)

    def apply(self, dataset) -> "TaskLabels":
        """`dataset` (anything with .latent_idx) relabelled by this task; images are shared."""
        n_factors = np.asarray(dataset.latent_idx).shape[1]
        if max(self.relevant_factors) >= n_factors:
            raise ValueError(f"factors {self.relevant_factors}: the dataset has factors 0-{n_factors - 1}")
        return TaskLabels(dataset, self.labels(dataset.latent_idx), self.n_classes)


class TwoFeatureParityTask(FeatureParityTask):
    """A FeatureParityTask over exactly two distinct features."""

    def __init__(self, f1: int, f2: int):
        super().__init__([f1, f2])


class TaskLabels(Dataset):
    """A dataset's inputs paired with the labels of a task."""

    def __init__(self, dataset, labels: np.ndarray, n_classes: int):
        self.dataset = dataset
        self.labels = labels
        self.n_classes = n_classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int):
        x, _ = self.dataset[i]
        return x, int(self.labels[i])




class FeatureSumTask(Task):
    """
    Binary task "sum above median" (notes/sum_above_median_labels.md): each feature's
    value index is scaled to [0, 1] (index / (n_values - 1)), the scaled values of
    `factors` are summed, and the upper half of all value combinations by that score
    is class 1, the lower half class 0.

    Shapes3D is a full factorial grid, so every combination of the task's features is
    equally frequent and splitting the combinations in half balances the classes.
    Combinations tied at the median are ordered by a fixed pseudo-random key, so the
    split is exactly half (one combination off when their number is odd).

    Shared features push both labels the same way, so tasks sharing features agree on
    more images the more they share -- unlike FeatureParityTask.
    """

    def __init__(self, factors, tie_seed: int = 0):
        factors = [int(f) for f in factors]
        if not factors or len(set(factors)) != len(factors) or not all(0 <= f < N_FACTORS for f in factors):
            raise ValueError(f"factors must be distinct Shapes3D factor indices 0-{N_FACTORS - 1}, got {factors}")
        shape = [FACTOR_N_VALUES[f] for f in factors]
        grids = np.meshgrid(*[np.arange(n) / (n - 1) for n in shape], indexing="ij")
        score = np.round(sum(grids).ravel(), 9)               # equal sums must compare equal
        tie = np.random.default_rng(tie_seed).random(score.size)
        order = np.lexsort((tie, score))                        # ascending score, ties by the fixed key
        table = np.zeros(score.size, dtype=np.int64)
        table[order[(score.size + 1) // 2:]] = 1                 # upper half -> class 1
        self.table = table.reshape(shape)
        self.median_score = float(np.median(score))
        super().__init__(relevant_factors=factors, label_fn=self._label,
                         name="sum(" + "+".join(FACTOR_NAMES[f] for f in factors) + ")", n_classes=2)

    def _label(self, latent_idx: np.ndarray) -> int:
        return int(self.table[tuple(int(latent_idx[f]) for f in self.relevant_factors)])

    def labels(self, latent_idx: np.ndarray) -> np.ndarray:
        """Vectorised labels for an (N, 6) array of latent indices."""
        li = np.asarray(latent_idx)
        return self.table[tuple(li[:, f] for f in self.relevant_factors)].astype(np.int64)

    def apply(self, dataset) -> "TaskLabels":
        """`dataset` (anything with .latent_idx) relabelled by this task; images are shared."""
        return TaskLabels(dataset, self.labels(dataset.latent_idx), self.n_classes)


# label rules for feature-set tasks, by name
SET_RULES = {"parity": FeatureParityTask, "sum": FeatureSumTask}