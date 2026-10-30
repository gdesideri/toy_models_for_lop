"""
src/unit_geometry.py — Geometry of single ReLU units w·x + b on a bounded input
support S = {μ + A u : ||u|| <= 1}: signed angles, the exact support extrema, and
certified (population-level) inactivity.

    max_{x in S} w·x + b = w·μ + ||Aᵀw|| + b        (attained at u = Aᵀw / ||Aᵀw||)
    min_{x in S} w·x + b = w·μ − ||Aᵀw|| + b

A unit is *certified inactive* when the maximum is <= 0: it is zero on the whole
support, gets no gradient from any input, and (with no bias, plain SGD) can never
come back. This is a statement about the population. A unit that happens to be
zero on a finite sample is only *silent on that sample*.

Weight vectors shorter than `eps` are *collapsed*: their direction is not defined,
so no angle is assigned (NaN) and they are not counted as rotated into the cone.

Angles are never folded with abs(dot): w and −w are 180° apart.
"""
from __future__ import annotations

import numpy as np

ACTIVE, INACTIVE, COLLAPSED = 0, 1, 2
STATUS_NAMES = ("active", "certified_inactive", "collapsed")

DEFAULT_EPS = 1e-3


def _rows(W) -> np.ndarray:
    return np.asarray(W, dtype=float)


# ── Support extrema and inactivity ────────────────────────────────────────────

def support_max(W, b, mu, A) -> np.ndarray:
    """max over the support of w·x + b, for every row w of W (..., D)."""
    W = _rows(W)
    return W @ np.asarray(mu, float) + np.linalg.norm(W @ np.asarray(A, float), axis=-1) + b


def support_min(W, b, mu, A) -> np.ndarray:
    """min over the support of w·x + b."""
    W = _rows(W)
    return W @ np.asarray(mu, float) - np.linalg.norm(W @ np.asarray(A, float), axis=-1) + b


def certified_inactive(W, b, mu, A) -> np.ndarray:
    """True where w·μ + ||Aᵀw|| + b <= 0: the unit is zero on the whole support."""
    return support_max(W, b, mu, A) <= 0


def directional_margin(W, mu, A, eps: float = DEFAULT_EPS) -> np.ndarray:
    """
    Support maximum of the unit direction ŵ with no bias, ŵ·μ + ||Aᵀŵ||.
    <= 0 exactly when the direction lies in the inactive cone; NaN for ||w|| < eps.
    """
    W = _rows(W)
    norm = np.linalg.norm(W, axis=-1)
    safe = np.where(norm < eps, 1.0, norm)
    m = support_max(W / safe[..., None], 0.0, mu, A)
    return np.where(norm < eps, np.nan, m)


def unit_status(W, b, mu, A, eps: float = DEFAULT_EPS) -> np.ndarray:
    """ACTIVE / INACTIVE (certified) / COLLAPSED (||w|| < eps) for every row of W."""
    W = _rows(W)
    norm = np.linalg.norm(W, axis=-1)
    status = np.where(certified_inactive(W, b, mu, A), INACTIVE, ACTIVE)
    return np.where(norm < eps, COLLAPSED, status).astype(np.int64)


def silent_on_sample(W, b, X) -> np.ndarray:
    """True where the unit's preactivation is <= 0 on every row of the finite sample X."""
    pre = np.asarray(X, float) @ _rows(W).T + b
    return pre.max(axis=0) <= 0


def inactive_arcs_2d(mu, A, b: float = 0.0, n_grid: int = 7200, tol: float = 1e-9):
    """
    Directions θ (degrees, in [-180, 180)) of the unit vectors (cos θ, sin θ) that
    are certified inactive with bias b, as a list of (start, end) arcs going
    counter-clockwise; an arc may wrap through ±180 (then start > end). Boundaries
    are refined by bisection. Empty when no direction is inactive (e.g. μ = 0, b = 0).
    """
    def margin(theta):
        theta = np.asarray(theta, float)
        return support_max(np.stack([np.cos(theta), np.sin(theta)], -1), b, mu, A)

    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    dead = margin(grid) <= 0
    if not dead.any():
        return []
    if dead.all():
        return [(-180.0, 180.0)]

    def refine(lo, hi, dead_at_hi):
        # boundary between lo and hi; dead_at_hi says which side is inactive
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if (margin(mid) <= 0) == dead_at_hi:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    step = grid[1] - grid[0]
    starts, ends = [], []
    for i in range(n_grid):
        j = (i + 1) % n_grid
        lo = grid[i]
        hi = lo + step
        if not dead[i] and dead[j]:
            starts.append(refine(lo, hi, True))
        if dead[i] and not dead[j]:
            ends.append(refine(lo, hi, False))

    def wrap(t):
        return float((np.degrees(t) + 180.0) % 360.0 - 180.0)

    starts, ends = sorted(wrap(s) for s in starts), [wrap(e) for e in ends]
    arcs = []
    for s in starts:
        # the arc ends at the first end reached counter-clockwise from s
        e = min(ends, key=lambda e: (e - s) % 360.0)
        arcs.append((s, e))
    return arcs


# ── Angles ────────────────────────────────────────────────────────────────────

def signed_angle_2d(W, eps: float = DEFAULT_EPS) -> np.ndarray:
    """atan2(w₂, w₁) in degrees, in (-180, 180]; NaN for ||w|| < eps."""
    W = _rows(W)
    ang = np.degrees(np.arctan2(W[..., 1], W[..., 0]))
    return np.where(np.linalg.norm(W, axis=-1) < eps, np.nan, ang)


def plane_angle(W, v_i, v_j, eps: float = DEFAULT_EPS) -> np.ndarray:
    """
    Signed angle (degrees) of the projection of w onto the plane spanned by the
    orthonormal v_i, v_j: atan2(w·v_j, w·v_i). NaN when the projection is shorter
    than eps. A projection: it says nothing about inactivity in the full space.
    """
    W = _rows(W)
    p_i, p_j = W @ np.asarray(v_i, float), W @ np.asarray(v_j, float)
    ang = np.degrees(np.arctan2(p_j, p_i))
    return np.where(np.hypot(p_i, p_j) < eps, np.nan, ang)


def angle_between(W, v, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Angle in [0, 180] degrees between w and v (signed cosine, no abs); NaN for short vectors."""
    W, v = _rows(W), np.asarray(v, float)
    nw = np.linalg.norm(W, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    cos = (W * v).sum(-1) / np.where((nw < eps) | (nv < eps), 1.0, nw * nv)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.where((nw < eps) | (nv < eps), np.nan, ang)


def radial_tangential(W, dW, eps: float = DEFAULT_EPS):
    """
    Split a change dW of each row w into its radial part dw·ŵ (> 0: w grows) and
    the length of its tangential part (rotation). NaN for ||w|| < eps.
    """
    W, dW = _rows(W), _rows(dW)
    norm = np.linalg.norm(W, axis=-1)
    unit = W / np.where(norm < eps, 1.0, norm)[..., None]
    radial = (dW * unit).sum(-1)
    tangential = np.linalg.norm(dW - radial[..., None] * unit, axis=-1)
    small = norm < eps
    return np.where(small, np.nan, radial), np.where(small, np.nan, tangential)
