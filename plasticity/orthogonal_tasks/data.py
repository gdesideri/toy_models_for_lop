"""
The input cloud and the tasks.

Inputs  x = A u, with u uniform inside the D-dimensional unit ball and A = diag(scales). The default
        is 1 on every axis and 0.5 on the last one (D = 2: an ellipse half as tall as wide). The cloud
        is centred at the origin.
Tasks   K tasks, each with a unit normal v_j (the hyperplane of the task):
          axes        the first K coordinate axes
          diagonal    K = 2: (e_1 + e_2)/√2 and (e_1 − e_2)/√2, each label uses two coordinates
          diagonal-c  K = 3: the two diagonal normals plus C = (e_1 + e_3)/√2 (at 60° to A and B)
Labels  halfspace  y = 1[v_j · x > 0]
        stripe     y = 1[|v_j · x| < h_j], a band around the centre with h_j the median of |v_j · x| on
                   the train set (balanced labels); one ReLU unit cannot fit it, it needs one per edge
        xor        K = 2: y = 1[(p_j · u)(q_j · u) > 0] on the round coordinates u, with (p_j, q_j) the
                   first two axes turned by j·45°: task A is the quadrant XOR, task B the same turned
                   45°; balanced, uncorrelated, and each needs 4 ReLU units
Data    one train set and one independent test set shared by every task: only the labels differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NORMAL_KINDS = ("axes", "diagonal", "diagonal-c")
LABEL_RULES = ("halfspace", "stripe", "xor")


def resolve_input_dim(n_tasks: int, input_dim: int | None = None) -> int:
    """D = max(2, K) when not given; D < K is rejected."""
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if input_dim is None:
        return max(2, n_tasks)
    if input_dim < n_tasks:
        raise ValueError(f"input_dim D={input_dim} < n_tasks K={n_tasks}: K task normals need at least K dims")
    return int(input_dim)


def default_scales(input_dim: int) -> np.ndarray:
    """1 on every axis, 0.5 on the last: (1, 0.5) in D = 2."""
    scales = np.ones(input_dim)
    scales[-1] = 0.5
    return scales


def task_normals(n_tasks: int, input_dim: int, kind: str = "axes") -> np.ndarray:
    """(K, D) unit task normals (see the module docstring)."""
    e = np.eye(input_dim)
    if kind == "axes":
        return e[:n_tasks].copy()
    if kind == "diagonal":
        if n_tasks != 2:
            raise ValueError(f"diagonal task normals need n_tasks = 2, got {n_tasks}")
        return np.stack([e[0] + e[1], e[0] - e[1]]) / np.sqrt(2.0)
    if kind == "diagonal-c":
        if n_tasks != 3 or input_dim < 3:
            raise ValueError(f"diagonal-c task normals need n_tasks = 3 and D >= 3, got {n_tasks}, {input_dim}")
        return np.stack([e[0] + e[1], e[0] - e[1], e[0] + e[2]]) / np.sqrt(2.0)
    raise ValueError(f"unknown task normals {kind!r}; choose from {NORMAL_KINDS}")


def sample_unit_ball(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """(n, dim) points uniform inside the unit ball: a uniform direction times radius U^(1/dim)."""
    g = rng.standard_normal((n, dim))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g * (rng.random(n) ** (1.0 / dim))[:, None]


def xor_pairs(n_tasks: int, input_dim: int) -> np.ndarray:
    """(K, 2, D) the directions (p_j, q_j) of each XOR task: e_1, e_2 turned by j·45° in their plane."""
    if n_tasks != 2:
        raise ValueError(f"xor labels need n_tasks = 2, got {n_tasks}")
    e = np.eye(input_dim)
    th = np.arange(n_tasks) * np.pi / 4
    return np.array([[np.cos(t) * e[0] + np.sin(t) * e[1], -np.sin(t) * e[0] + np.cos(t) * e[1]] for t in th])


def halfspace_labels(x: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """(K, n) labels 1[v_j · x > 0]."""
    return (x @ normals.T > 0).T.astype(np.int64)


def stripe_labels(x: np.ndarray, normals: np.ndarray, half_widths: np.ndarray) -> np.ndarray:
    """(K, n) labels 1[|v_j · x| < h_j]."""
    return (np.abs(x @ normals.T) < half_widths).T.astype(np.int64)


def stripe_half_widths(x_train: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """(K,) half-widths putting half of the train points inside each stripe: the median of |v_j · x|."""
    return np.median(np.abs(x_train @ normals.T), axis=0)


def xor_labels(u: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """(K, n) labels 1[(p_j · u)(q_j · u) > 0] of round coordinates u."""
    return np.stack([((u @ p) * (u @ q) > 0).astype(np.int64) for p, q in pairs])


def task_schedule(n_tasks: int, phases: int) -> list[int]:
    """Task of every phase: 0, 1, ..., K−1, 0, 1, ... for `phases` phases."""
    return [p % n_tasks for p in range(phases)]


@dataclass
class Tasks:
    """The fixed data of one seed: ball samples u, the shape A, the task normals and the labels."""
    normals: np.ndarray            # (K, D)
    A: np.ndarray                  # (D, D)
    u_train: np.ndarray            # (n_train, D)
    u_test: np.ndarray             # (n_test, D)
    label_rule: str = "halfspace"
    half_widths: np.ndarray | None = field(init=False)   # (K,) stripes only
    pairs: np.ndarray | None = field(init=False)         # (K, 2, D) XOR only
    labels_train: np.ndarray = field(init=False)         # (K, n_train)
    labels_test: np.ndarray = field(init=False)          # (K, n_test)

    def __post_init__(self):
        if self.label_rule not in LABEL_RULES:
            raise ValueError(f"unknown label rule {self.label_rule!r}; choose from {LABEL_RULES}")
        self.half_widths, self.pairs = None, None
        if self.label_rule == "halfspace":
            label = lambda split: halfspace_labels(self.inputs(split), self.normals)  # noqa: E731
        elif self.label_rule == "stripe":
            self.half_widths = stripe_half_widths(self.inputs("train"), self.normals)
            label = lambda split: stripe_labels(self.inputs(split), self.normals, self.half_widths)  # noqa: E731
        else:
            self.pairs = xor_pairs(self.n_tasks, self.input_dim)
            label = lambda split: xor_labels(self.u(split), self.pairs)  # noqa: E731
        self.labels_train, self.labels_test = label("train"), label("test")

    @property
    def n_tasks(self) -> int:
        return self.normals.shape[0]

    @property
    def input_dim(self) -> int:
        return self.normals.shape[1]

    def u(self, split: str) -> np.ndarray:
        return self.u_train if split == "train" else self.u_test

    def inputs(self, split: str) -> np.ndarray:
        """(n, D) inputs x = A u of a split."""
        return self.u(split) @ self.A.T

    def boundaries(self, task: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """The task's decision boundaries as (point, unit normal) hyperplanes in input space: one for a
        half-plane, the two edges of a stripe, the two lines of an XOR (p·u = 0 and q·u = 0)."""
        v = self.normals[task]
        if self.label_rule == "stripe":
            h = self.half_widths[task]
            return [(-h * v, v), (h * v, v)]
        if self.label_rule == "xor":
            Ainv_T = np.linalg.inv(self.A).T          # p·u = p·A⁻¹x = (A⁻ᵀp)·x
            out = []
            for d in self.pairs[task]:
                n = Ainv_T @ d
                out.append((np.zeros_like(v), n / np.linalg.norm(n)))
            return out
        return [(np.zeros_like(v), v)]


def build_tasks(n_tasks: int = 2, input_dim: int | None = None, n_train: int = 2048, n_test: int = 2048,
                seed: int = 0, scales=None, normals: str = "axes", label_rule: str = "halfspace") -> Tasks:
    """The task family with its fixed train and test samples, drawn from `seed`."""
    d = resolve_input_dim(n_tasks, input_dim)
    scales = default_scales(d) if scales is None else np.asarray(scales, dtype=float)
    if scales.shape != (d,) or np.any(scales <= 0):
        raise ValueError(f"scales must be {d} positive numbers, got {scales}")
    rng = np.random.default_rng(seed)
    u_train = sample_unit_ball(n_train, d, rng)
    u_test = sample_unit_ball(n_test, d, rng)
    return Tasks(task_normals(n_tasks, d, normals), np.diag(scales), u_train, u_test, label_rule)
