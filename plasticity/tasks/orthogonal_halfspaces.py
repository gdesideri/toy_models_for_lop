"""
tasks/orthogonal_halfspaces.py — K binary tasks with orthogonal decision hyperplanes
on one fixed, bounded input cloud (the orthogonal-tasks toy experiment, run by
experiments/run_orthogonal_tasks.py).

Inputs   x = μ + A u, with u uniform inside the D-dimensional unit ball and A a
         shape matrix (diagonal with positive scales by default). The default in
         D = 2 is the shifted ellipse μ = (0, 1), A = diag(1, 0.5). In D > 2 the
         shift stays on the last axis: μ = e_{D-1}, scale 0.5 on that axis and 1
         on the others. Both are overridable.
Tasks    K known orthonormal normals v_0 .. v_{K-1}: the first K coordinate axes
         (no PCA), or with normals="diagonal" (K = 2) (e_0 ± e_1)/√2.
Labels   label_rule="halfspace": y = 1[v_j · (x − μ) > 0]. label_rule="stripe": y = 1[|v_j · (x − μ)| < h_j],
         a band around the centre with h_j the median of |v_j · (x − μ)| on the train set (balanced labels);
         one ReLU unit cannot represent a stripe even with a bias, it needs one unit per edge.
         label_rule="xor" (K = 2): y = 1[(p_j · u)(q_j · u) > 0] on the round coordinates u = A⁻¹(x − μ), with
         (p_j, q_j) the first two axes turned by j·45°: task A is the quadrant XOR, task B the same XOR turned
         45° (|u_1| > |u_0|). Labels are balanced and A, B uncorrelated; each needs 4 ReLU units.
         D must be at least K; D defaults to max(2, K).
Data     One fixed train set and one independent test set, shared by every task:
         only the labels differ between tasks. Labels are computed from
         z = x − μ = A u, so the centred control (μ = 0, the same u) has exactly
         the same labels.
Stream   Tasks 0, 1, ..., K−1, 0, 1, ... (`task_schedule`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Dimensions and task normals ───────────────────────────────────────────────

def resolve_input_dim(n_tasks: int, input_dim: int | None = None) -> int:
    """D = max(2, K) when not given; D < K is rejected (K orthogonal normals need K dims)."""
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if input_dim is None:
        return max(2, n_tasks)
    if input_dim < n_tasks:
        raise ValueError(
            f"input_dim D={input_dim} < n_tasks K={n_tasks}: K orthogonal task normals "
            "need at least K input dimensions")
    return int(input_dim)


NORMAL_KINDS = ("axes", "diagonal", "diagonal-c")


def task_normals(n_tasks: int, input_dim: int, kind: str = "axes") -> np.ndarray:
    """(K, D) orthonormal task normals.

    axes      the first K coordinate axes
    diagonal    K = 2 only: (e_0 + e_1)/√2 and (e_0 − e_1)/√2, so each label depends on two coordinates
    diagonal-c  K = 3, D >= 3: the two diagonal normals plus C = (e_0 + e_2)/√2, which uses x_2 (the third
                coordinate); C is at 60° to A and B, so this family is NOT orthogonal
    """
    input_dim = resolve_input_dim(n_tasks, input_dim)
    if kind == "axes":
        return np.eye(input_dim)[:n_tasks].copy()
    if kind == "diagonal":
        if n_tasks != 2:
            raise ValueError(f"diagonal task normals need n_tasks = 2, got {n_tasks}")
        e = np.eye(input_dim)
        return np.stack([e[0] + e[1], e[0] - e[1]]) / np.sqrt(2.0)
    if kind == "diagonal-c":
        if n_tasks != 3:
            raise ValueError(f"diagonal-c task normals need n_tasks = 3, got {n_tasks}")
        e = np.eye(input_dim)
        return np.stack([e[0] + e[1], e[0] - e[1], e[0] + e[2]]) / np.sqrt(2.0)
    raise ValueError(f"unknown task normals {kind!r}; choose from {NORMAL_KINDS}")


# ── Input distribution ────────────────────────────────────────────────────────

def default_mu(input_dim: int) -> np.ndarray:
    """The shift of the shifted condition: e_{D-1}, i.e. (0, 1) in D = 2."""
    mu = np.zeros(input_dim)
    mu[-1] = 1.0
    return mu


def default_scales(input_dim: int) -> np.ndarray:
    """Diagonal of A: 1 everywhere, 0.5 on the shift axis, i.e. (1, 0.5) in D = 2."""
    scales = np.ones(input_dim)
    scales[-1] = 0.5
    return scales


def shape_matrix(scales) -> np.ndarray:
    """A = diag(scales); every scale must be positive."""
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or np.any(scales <= 0):
        raise ValueError(f"scales must be a vector of positive numbers, got {scales}")
    return np.diag(scales)


def sample_unit_ball(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """(n, dim) points uniform inside the unit ball: a uniform direction times radius U^(1/dim)."""
    g = rng.standard_normal((n, dim))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    r = rng.random(n) ** (1.0 / dim)
    return g * r[:, None]


LABEL_RULES = ("halfspace", "stripe", "xor")


def task_labels(x: np.ndarray, mu: np.ndarray, normals: np.ndarray, half_widths=None) -> np.ndarray:
    """(K, n) labels of inputs x (n, D): 1[v_j · (x − μ) > 0] (half-space), or, when half_widths is given,
    1[|v_j · (x − μ)| < h_j] (a stripe of half-width h_j around the centre, which no single ReLU unit can
    represent: it needs one unit per edge)."""
    proj = (np.asarray(x) - mu) @ normals.T
    if half_widths is None:
        return (proj > 0).T.astype(np.int64)
    return (np.abs(proj) < np.asarray(half_widths)).T.astype(np.int64)


def stripe_half_widths(z_train: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """(K,) half-widths that put half the train points inside each stripe: the median of |v_j · z|."""
    return np.median(np.abs(z_train @ normals.T), axis=0)


def xor_pairs(n_tasks: int, input_dim: int) -> np.ndarray:
    """(K, 2, D) the two directions (p_j, q_j) of each XOR task: e_0, e_1 turned by j·45° in their plane."""
    if n_tasks != 2:
        raise ValueError(f"xor labels need n_tasks = 2, got {n_tasks}")
    e = np.eye(input_dim)
    out = []
    for j in range(n_tasks):
        th = j * np.pi / 4
        out.append([np.cos(th) * e[0] + np.sin(th) * e[1], -np.sin(th) * e[0] + np.cos(th) * e[1]])
    return np.array(out)


def xor_labels(u: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """(K, n) XOR labels 1[(p_j · u)(q_j · u) > 0] of round coordinates u (n, D)."""
    return np.stack([((u @ p) * (u @ q) > 0).astype(np.int64) for p, q in pairs])


def task_schedule(n_tasks: int, cycles: int) -> list[int]:
    """Task of every phase: 0, 1, ..., K−1 repeated `cycles` times."""
    return [j for _ in range(cycles) for j in range(n_tasks)]


# ── The task family ───────────────────────────────────────────────────────────

CONDITIONS = ("shifted", "centered")


@dataclass
class OrthogonalTasks:
    """
    The fixed data of one seed: unit-ball samples u, the geometry (μ, A) and the
    K orthogonal tasks. Inputs of a condition are μ_condition + A u; the labels
    belong to u alone and are shared by both conditions.
    """
    normals: np.ndarray   # (K, D)
    mu: np.ndarray        # (D,) mean of the shifted condition
    A: np.ndarray         # (D, D)
    u_train: np.ndarray   # (n_train, D)
    u_test: np.ndarray    # (n_test, D)
    label_rule: str = "halfspace"                  # "halfspace" | "stripe" | "xor"
    half_widths: np.ndarray | None = field(init=False)   # (K,) stripe half-widths, None for half-spaces
    pairs: np.ndarray | None = field(init=False)         # (K, 2, D) XOR directions in u, None otherwise
    labels_train: np.ndarray = field(init=False)   # (K, n_train)
    labels_test: np.ndarray = field(init=False)    # (K, n_test)

    def __post_init__(self):
        if self.label_rule not in LABEL_RULES:
            raise ValueError(f"unknown label rule {self.label_rule!r}; choose from {LABEL_RULES}")
        zero = np.zeros_like(self.mu)
        self.pairs = xor_pairs(self.n_tasks, self.input_dim) if self.label_rule == "xor" else None
        self.half_widths = (stripe_half_widths(self.offsets("train"), self.normals)
                            if self.label_rule == "stripe" else None)
        if self.pairs is not None:
            self.labels_train = xor_labels(self.u_train, self.pairs)
            self.labels_test = xor_labels(self.u_test, self.pairs)
            return
        self.labels_train = task_labels(self.offsets("train"), zero, self.normals, self.half_widths)
        self.labels_test = task_labels(self.offsets("test"), zero, self.normals, self.half_widths)

    @property
    def n_tasks(self) -> int:
        return self.normals.shape[0]

    @property
    def input_dim(self) -> int:
        return self.normals.shape[1]

    def offsets(self, split: str) -> np.ndarray:
        """z = A u = x − μ of a split, the same in every condition."""
        u = self.u_train if split == "train" else self.u_test
        return u @ self.A.T

    def condition_mu(self, condition: str) -> np.ndarray:
        if condition == "shifted":
            return self.mu.copy()
        if condition == "centered":
            return np.zeros_like(self.mu)
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")

    def inputs(self, split: str, condition: str = "shifted") -> np.ndarray:
        """(n, D) inputs x = μ_condition + A u of a split."""
        return self.condition_mu(condition) + self.offsets(split)


def build_tasks(n_tasks: int = 2, input_dim: int | None = None, n_train: int = 2048,
                n_test: int = 2048, seed: int = 0, mu=None, scales=None,
                normals: str = "axes", label_rule: str = "halfspace") -> OrthogonalTasks:
    """The K-task family with its fixed train and test samples, drawn from `seed`."""
    d = resolve_input_dim(n_tasks, input_dim)
    mu = default_mu(d) if mu is None else np.asarray(mu, dtype=float)
    scales = default_scales(d) if scales is None else np.asarray(scales, dtype=float)
    if mu.shape != (d,) or scales.shape != (d,):
        raise ValueError(f"mu and scales must have D={d} entries, got {mu.shape} and {scales.shape}")
    rng = np.random.default_rng(seed)
    u_train = sample_unit_ball(n_train, d, rng)
    u_test = sample_unit_ball(n_test, d, rng)
    return OrthogonalTasks(task_normals(n_tasks, d, normals), mu, shape_matrix(scales), u_train, u_test,
                           label_rule)
