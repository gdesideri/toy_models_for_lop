"""
experiments/run_orthogonal_tasks.py — Continual learning on K binary tasks with
orthogonal decision hyperplanes, tracking every first-layer ReLU unit.

Question: does adapting between orthogonal tasks rotate initially active hidden
units into the inactive cone of weight space (directions whose preactivation is
<= 0 on the whole bounded input support)? Not seeing it is a valid result.

Setup (tasks/orthogonal_halfspaces.py)
    x = μ + A u, u uniform in the D-ball; default D = 2: μ = (0, 1), A = diag(1, 0.5).
    Task j: y = 1[v_j · (x − μ) > 0] (`--label-rule stripe`: y = 1[|v_j · (x − μ)| < h_j], a
    balanced band that needs two units; `--label-rule xor` (K = 2): task A is the quadrant XOR and task B the
    same XOR turned 45°, both on the round coordinates u, 4 units each), v_j = e_j (orthonormal, no PCA); with
    `--task-normals diagonal` (K = 2) v = (e_0 ± e_1)/√2, two coordinates per label;
    `--task-normals diagonal-c` (K = 3) adds C = (e_0 + e_2)/√2 (at 60° to A and B).
    Phases cycle 0, 1, ..., K−1, 0, 1, ... (`--cycles` passes, or `--phases` phases in
    total, `--steps-per-task` each).
    One fixed train set and an independent test set shared by all tasks.
    Conditions: "shifted" (μ as above) and the "centered" control (μ = 0, the same
    u, labels and seeds). With zero hidden bias the centered control has no
    nonzero inactive direction.
Network
    D → width ReLU units (hidden bias: none by default, `--hidden-bias trainable`)
    → 1 logit with a trainable output bias; BCE with logits; plain SGD (momentum 0)
    by default. Network and optimizer state carry across task switches. Both
    layers train. Built with models.factory ("shallow_<width>"), float64.
    --init kaiming (default): the factory's uniform Kaiming init, so some units can
    start silent (in the shifted condition ~23% of directions are in the cone).
    --init live: every hidden unit that is silent on the shifted train inputs is
    redrawn from the same uniform Kaiming distribution until it fires on some of them
    (rejection sampling, so live units keep their distribution; units already live
    are untouched). Used for the continued net and every fresh net, and the same
    weights start both conditions.
Fresh baseline
    At every phase a freshly initialised network trains next to the continued one
    on the identical minibatches. Fresh networks must solve every task (final test
    accuracy >= --solve-threshold) before a continued-vs-fresh gap is read as loss
    of plasticity; the run reports this per task.
Tracked per hidden unit (every --log-every steps and at every switch)
    input weights w, norm, 2-D signed angle atan2(w₂, w₁) (no abs: w and −w differ),
    angle to the current task normal, directional margin ŵ·μ + ||Aᵀŵ||, support
    max/min of the preactivation, certified status (active / certified inactive /
    collapsed ||w|| < --norm-eps), activation frequency and preactivation extrema
    on the train and test samples, output weight a (signed), the unit's output
    contribution and an output-cancellation index, full-batch train gradients for
    every task (exact, analytic), the minibatch gradient and the parameter update
    of the last step (radial = norm change, tangential = rotation).
    Exact per step: status changes, silence on the train sample, readout sign
    flips, path length and angle travelled.
    Death of an initially active unit = the first step at which it is silent on
    the whole train sample (zero training gradient; permanent under plain SGD with
    a fixed train set) or certified inactive / collapsed. Mechanisms:
      rotation_into_cone     direction inside the certified cone (certified)
      rotation_to_cone_edge  silent on every training input but NOT certified: the
                             direction stops in the thin sliver outside the cone
                             where the support is still positive but no sample is
      norm_collapse          ||w|| < eps
      bias_driven            the direction alone is active, a negative hidden bias
                             silences it (only with --hidden-bias trainable)

Usage (from plasticity/)
    uv run python -m experiments.run_orthogonal_tasks --seeds 0 --steps-per-task 500 --cycles 2   # smoke
    uv run python -m experiments.run_orthogonal_tasks --seeds 0 1 2 3 4 5 6 7 8 9                 # default (~30 s/seed)
    uv run python -m experiments.run_orthogonal_tasks --width 16 --seeds 0 1 2 3 4 5 6 7 8 9      # fresh nets solvable
    uv run python -m experiments.run_orthogonal_tasks --n-tasks 3 --seeds 0 1 2 3 4                # D = 3
    uv run python -m experiments.run_orthogonal_tasks --n-tasks 2 --input-dim 4 --seeds 0 1 2
    uv run python -m experiments.run_orthogonal_tasks --init live --seeds 0 1 2 3 4 5 6 7 8 9    # no unit starts silent
    then: uv run python -m visualization_scripts.toy_models.plot_orthogonal_tasks --run-dir <printed dir>

Outputs: results/orthogonal_tasks/<run name>/
    seed<s>/config.json      the resolved config
    seed<s>/<condition>.npz  time series of the continued net (keys without prefix),
                             of the fresh nets (prefix fresh_, with fresh_phase),
                             per-step events (events_*, fresh_events_*), geometry
    seed<s>/summary.json     per unit, per phase and per condition summaries
    aggregate.json           counts over all seeds in the folder (every seed, no selection)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.factory import build_model, build_optimizer
from src import unit_geometry as geo
from tasks.orthogonal_halfspaces import (CONDITIONS, LABEL_RULES, NORMAL_KINDS, OrthogonalTasks, build_tasks, default_mu,
                                         default_scales, resolve_input_dim, task_schedule)

RESULTS = Path(__file__).resolve().parent.parent / "results" / "orthogonal_tasks"
DTYPE = torch.float64
# How an initially active unit stopped receiving gradient (see `deaths`).
MECHANISMS = ("rotation_into_cone", "rotation_to_cone_edge", "norm_collapse", "bias_driven")
EVENT_STATUS, EVENT_SILENT, EVENT_SIGN = 0, 1, 2
EVENT_INFO = ("norm", "dir_margin", "bias_over_norm", "angle_2d", "support_max",
              "train_dir_max", "test_act_frac")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    n_tasks: int = 2
    input_dim: int | None = None          # None: max(2, n_tasks)
    mu: list[float] | None = None         # None: e_{D-1}, (0, 1) in D = 2
    scales: list[float] | None = None     # None: 1 everywhere, 0.5 on the last axis
    width: int = 4
    hidden_bias: str = "zero"             # "zero" (no hidden bias) | "trainable"
    init: str = "kaiming"                 # "kaiming" | "live" (redraw units silent at init)
    bias_init: str = "zero"               # hidden bias at init: "zero" | "uniform" U(-1/sqrt(D), 1/sqrt(D))
    optimizer: str = "sgd"
    momentum: float = 0.0
    lr: float = 0.1
    batch_size: int = 64
    steps_per_task: int = 4000
    cycles: int = 4
    label_rule: str = "halfspace"         # "halfspace" | "stripe" (needs >= 2 units per task)
    task_normals: str = "axes"            # "axes" | "diagonal" (K = 2: (e_0 ± e_1)/√2)
    phases: int | None = None             # None: K * cycles; else the first `phases` phases
    n_train: int = 2048
    n_test: int = 2048
    log_every: int = 10
    norm_eps: float = geo.DEFAULT_EPS
    solve_threshold: float = 0.95
    seed: int = 0
    conditions: list[str] = field(default_factory=lambda: list(CONDITIONS))

    def resolved(self) -> "Config":
        """A copy with D, μ and the scales filled in and every field validated."""
        d = resolve_input_dim(self.n_tasks, self.input_dim)
        mu = default_mu(d) if self.mu is None else np.asarray(self.mu, float)
        scales = default_scales(d) if self.scales is None else np.asarray(self.scales, float)
        if mu.shape != (d,) or scales.shape != (d,):
            raise ValueError(f"--mu and --scales need D={d} values each")
        if np.any(scales <= 0):
            raise ValueError("--scales must be positive")
        if self.hidden_bias not in ("zero", "trainable"):
            raise ValueError("--hidden-bias is 'zero' or 'trainable'")
        if self.init not in ("kaiming", "live"):
            raise ValueError("--init is 'kaiming' or 'live'")
        if self.bias_init not in ("zero", "uniform"):
            raise ValueError("--bias-init is 'zero' or 'uniform'")
        if self.bias_init == "uniform" and self.hidden_bias != "trainable":
            raise ValueError("--bias-init uniform needs --hidden-bias trainable")
        if self.width < 1 or self.steps_per_task < 1 or self.cycles < 1 or self.log_every < 1:
            raise ValueError("width, steps per task, cycles and log_every must be >= 1")
        if self.label_rule not in LABEL_RULES:
            raise ValueError(f"--label-rule is one of {LABEL_RULES}")
        if self.label_rule == "xor" and (self.n_tasks != 2 or self.task_normals != "axes"):
            raise ValueError("--label-rule xor needs --n-tasks 2 and --task-normals axes")
        if self.task_normals not in NORMAL_KINDS:
            raise ValueError(f"--task-normals is one of {NORMAL_KINDS}")
        if self.task_normals == "diagonal" and self.n_tasks != 2:
            raise ValueError("--task-normals diagonal needs --n-tasks 2")
        if self.task_normals == "diagonal-c" and self.n_tasks != 3:
            raise ValueError("--task-normals diagonal-c needs --n-tasks 3")
        if self.phases is not None and self.phases < 1:
            raise ValueError("--phases must be >= 1")
        for c in self.conditions:
            if c not in CONDITIONS:
                raise ValueError(f"unknown condition {c!r}; choose from {CONDITIONS}")
        out = Config(**asdict(self))
        out.input_dim, out.mu, out.scales = d, mu.tolist(), scales.tolist()
        return out

    @property
    def schedule(self) -> list[int]:
        if self.phases is None:
            return task_schedule(self.n_tasks, self.cycles)
        return task_schedule(self.n_tasks, -(-self.phases // self.n_tasks))[: self.phases]


def _axis_word(v, what: str) -> str:
    """'x<i>' when v is zero except on axis i (with the value when it is not 1), else the values."""
    nz = np.flatnonzero(np.abs(v) > 1e-12)
    if len(nz) == 1:
        val = v[nz[0]]
        return f"{what}-x{nz[0] + 1}" + ("" if np.isclose(val, 1) else f"by{val:g}")
    return f"{what}-" + ",".join(f"{x:g}" for x in v)


def run_name(cfg: Config) -> str:
    """Human-readable folder name of a configuration (seed excluded): every knob that changes the run,
    e.g. 2d_AB-axes_shift-x2_thin-x2_no-bias_4-units_5k-steps_10-switches_live-init. Defaults of the
    optimiser, learning rate, batch size and sample size are left out; other values are appended."""
    cfg = cfg.resolved()
    d, k = cfg.input_dim, cfg.n_tasks
    tasks = "".join(chr(ord("A") + j) for j in range(k)) if k <= 26 else f"{k}tasks"
    mu, scales = np.asarray(cfg.mu), np.asarray(cfg.scales)
    if not np.any(mu):
        shift = "no-shift"
    else:
        shift = _axis_word(mu, "shift")
    off = np.flatnonzero(~np.isclose(scales, 1))
    if len(off) == 0:
        cloud = "round-cloud"
    elif len(off) == 1 and np.isclose(scales[off[0]], 0.5):
        cloud = f"thin-x{off[0] + 1}"
    else:
        cloud = "scales-" + ",".join(f"{x:g}" for x in scales)
    S = cfg.steps_per_task
    steps = f"{S // 1000}k-steps" if S % 1000 == 0 else f"{S}-steps"
    normals = "diagonal" if cfg.task_normals == "diagonal-c" else cfg.task_normals   # ABC already names task C
    rule = {"stripe": "-stripes", "xor": "-xor"}.get(cfg.label_rule, "")
    parts = [f"{d}d", f"{tasks}-{normals}{rule}", shift, cloud,
             ("random-bias" if cfg.bias_init == "uniform" else "trainable-bias")
             if cfg.hidden_bias == "trainable" else "no-bias",
             f"{cfg.width}-units", steps, f"{len(cfg.schedule) - 1}-switches", f"{cfg.init}-init"]
    d0 = Config()
    if cfg.optimizer != d0.optimizer or cfg.momentum != d0.momentum:
        parts.append(cfg.optimizer + (f"-momentum{cfg.momentum:g}" if cfg.momentum else ""))
    if cfg.lr != d0.lr:
        parts.append(f"lr{cfg.lr:g}")
    if cfg.batch_size != d0.batch_size:
        parts.append(f"batch{cfg.batch_size}")
    if cfg.n_train != d0.n_train:
        parts.append(f"{cfg.n_train}-train")
    return "_".join(parts)


def seed_for(*keys) -> int:
    """A reproducible 32-bit seed from a tuple of ints and strings."""
    ints = [k if isinstance(k, int) else int.from_bytes(hashlib.sha256(k.encode()).digest()[:4], "little")
            for k in keys]
    return int(np.random.SeedSequence(ints).generate_state(1)[0])


# ── Network ───────────────────────────────────────────────────────────────────

def make_net(cfg: Config, seed: int, live_inputs: torch.Tensor | None = None) -> torch.nn.Module:
    """
    D → width ReLU → 1 logit with a trainable output bias, initialised from `seed`. With
    cfg.init == "live", hidden units silent on `live_inputs` are redrawn (see redraw_silent_units);
    the number of redraws is kept in `net.init_redraws`.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = build_model(f"shallow_{cfg.width}", d_in=cfg.input_dim,
                          hidden_bias=cfg.hidden_bias == "trainable", output_bias=True)
    net = net.to(DTYPE)
    if cfg.bias_init == "uniform":      # PyTorch's default bias range, drawn from its own stream
        g = torch.Generator().manual_seed(seed_for(seed, "bias-init"))
        bound = 1.0 / cfg.input_dim ** 0.5
        with torch.no_grad():
            net.fc1.bias.copy_((torch.rand(cfg.width, generator=g, dtype=DTYPE) * 2 - 1) * bound)
    net.init_redraws = 0
    if cfg.init == "live":
        if live_inputs is None:
            raise ValueError("init='live' needs the inputs the units must be active on")
        net.init_redraws = redraw_silent_units(net, live_inputs, seed)
    return net


def redraw_silent_units(net, X: torch.Tensor, seed: int, max_tries: int = 100_000) -> int:
    """
    Redraw every hidden unit whose preactivation is <= 0 on all rows of X from the same
    uniform Kaiming distribution as the factory (bound sqrt(6 / fan_in)) until it is
    positive on at least one row. Units already active are left as they are. Returns the
    number of redraws.
    """
    W, b = net.fc1.weight, net.fc1.bias
    bound = (6.0 / W.shape[1]) ** 0.5
    g = torch.Generator().manual_seed(seed_for(seed, "live-init"))
    redraws = 0
    with torch.no_grad():
        for h in range(W.shape[0]):
            bias = b[h] if b is not None else 0.0
            for _ in range(max_tries):
                if (X @ W[h] + bias).max() > 0:
                    break
                W[h] = (torch.rand(W.shape[1], generator=g, dtype=W.dtype) * 2 - 1) * bound
                redraws += 1
            else:
                raise RuntimeError(f"unit {h}: no active direction found in {max_tries} draws")
    return redraws


def make_optimizer(cfg: Config, net: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return build_optimizer(net, "sgd", lr=cfg.lr, momentum=cfg.momentum)
    return build_optimizer(net, cfg.optimizer, lr=cfg.lr)


def params(net):
    """(W (H, D), b (H,), a (H,), c ()) as detached tensors; b is zeros without a hidden bias."""
    W = net.fc1.weight.detach()
    b = net.fc1.bias.detach() if net.fc1.bias is not None else torch.zeros(W.shape[0], dtype=W.dtype)
    return W, b, net.fc2.weight.detach()[0], net.fc2.bias.detach()[0]


def full_batch_grads(W, b, a, c, X, Y):
    """
    Exact gradients of the mean BCE of every task (rows of Y, (K, N)) at (W, b, a, c),
    with ReLU'(0) = 0 as in autograd. Returns grad_W (K, H, D), grad_b (K, H),
    grad_a (K, H), grad_c (K,), loss (K,), acc (K,).
    """
    P = X @ W.T + b
    Hid = torch.relu(P)
    f = Hid @ a + c
    R = (torch.sigmoid(f)[None, :] - Y) / X.shape[0]            # dL/df per task
    M = (P > 0).to(X.dtype)
    grad_a = R @ Hid
    grad_c = R.sum(1)
    grad_b = a[None, :] * (R @ M)
    grad_W = a[None, :, None] * torch.einsum("kn,nh,nd->khd", R, M, X)
    loss = torch.stack([F.binary_cross_entropy_with_logits(f, y) for y in Y])
    acc = ((f[None, :] > 0).to(Y.dtype) == Y).to(X.dtype).mean(1)
    return grad_W, grad_b, grad_a, grad_c, loss, acc


# ── Batches ───────────────────────────────────────────────────────────────────

class BatchStream:
    """Minibatch indices of the fixed train set: reshuffled each epoch, from one seed."""

    def __init__(self, n: int, batch_size: int, seed: int):
        self.n, self.bs = n, batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.perm, self.pos = torch.randperm(n, generator=self.g), 0

    def next(self) -> torch.Tensor:
        if self.pos + self.bs > self.n:
            self.perm, self.pos = torch.randperm(self.n, generator=self.g), 0
        idx = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return idx


# ── Training step and per-step tracking ───────────────────────────────────────

def train_step(net, opt, xb, yb) -> dict:
    """One optimizer step; returns the minibatch gradient and the parameter update."""
    W0, b0, a0, _ = (t.clone() for t in params(net))
    loss = F.binary_cross_entropy_with_logits(net(xb), yb)
    opt.zero_grad()
    loss.backward()
    gW = net.fc1.weight.grad.detach().clone()
    ga = net.fc2.weight.grad.detach()[0].clone()
    opt.step()
    W1, b1, a1, _ = params(net)
    return {"loss": loss.item(), "grad_W": gW, "grad_a": ga,
            "W_prev": W0, "dW": W1 - W0, "da": a1 - a0, "db": b1 - b0}


class UnitTracker:
    """Exact per-step events of one network: status changes, train-sample silence, readout sign flips."""

    def __init__(self, net, X_train, X_test, mu, A, eps, step0: int):
        self.X, self.X_test, self.mu, self.A, self.eps = X_train, X_test, mu, A, eps
        W, b, a, _ = params(net)
        self.status = self._status(W, b)
        self.silent = self._silent(W, b)
        self.sign = np.sign(a.numpy())
        self.init_status, self.init_silent = self.status.copy(), self.silent.copy()
        self.events = []        # (step, unit, kind, old, new)
        self.event_info = []    # EVENT_INFO fields at the event
        self.path_length = np.zeros(len(self.status))
        self.angle_path = np.zeros(len(self.status))

    def _status(self, W, b):
        return geo.unit_status(W.numpy(), b.numpy(), self.mu, self.A, self.eps)

    def _silent(self, W, b):
        return ((self.X @ W.T + b).max(0).values <= 0).numpy()

    def update(self, net, step: int, rec: dict) -> None:
        W, b, a, _ = params(net)
        Wn, bn = W.numpy(), b.numpy()
        status, silent, sign = self._status(W, b), self._silent(W, b), np.sign(a.numpy())
        dW = rec["dW"].numpy()
        self.path_length += np.linalg.norm(dW, axis=1)
        self.angle_path += np.nan_to_num(geo.angle_between(rec["W_prev"].numpy(), Wn, self.eps))
        changed = [(EVENT_STATUS, self.status, status), (EVENT_SILENT, self.silent, silent),
                   (EVENT_SIGN, self.sign, sign)]
        for kind, old, new in changed:
            for h in np.flatnonzero(old != new):
                self.events.append((step, int(h), kind, int(old[h]), int(new[h])))
                self.event_info.append(self._info(W[h], b[h]))
        self.status, self.silent, self.sign = status, silent, sign

    def _info(self, w, b) -> tuple:
        wn, bn = w.numpy(), float(b)
        norm = float(np.linalg.norm(wn))
        unit = w / norm if norm >= self.eps else w * np.nan
        return (norm,
                float(geo.directional_margin(wn, self.mu, self.A, self.eps)),
                bn / norm if norm >= self.eps else np.nan,
                float(geo.signed_angle_2d(wn, self.eps)) if wn.shape[0] == 2 else np.nan,
                float(geo.support_max(wn, bn, self.mu, self.A)),
                float((self.X @ unit).max()),                     # max_n ŵ·x_n on the train sample
                float(((self.X_test @ w + b) > 0).double().mean()))

    def arrays(self, prefix: str = "") -> dict:
        ev = np.array(self.events, dtype=np.int64).reshape(-1, 5)
        info = np.array(self.event_info, dtype=float).reshape(-1, len(EVENT_INFO))
        return {f"{prefix}events": ev, f"{prefix}events_info": info}


# ── Snapshots ─────────────────────────────────────────────────────────────────

class Log:
    """Snapshots of one network (continued) or of the fresh networks (one per phase)."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, step, phase, task, snap, last) -> None:
        row = dict(snap, step=step, phase=phase, task=task)
        row.update(last_update_stats(snap["W"], last, snap["W"].shape[0]))
        self.rows.append(row)

    def arrays(self, prefix: str = "") -> dict:
        return {prefix + k: np.stack([np.asarray(r[k]) for r in self.rows]) for k in self.rows[0]}


def last_update_stats(W, last, width) -> dict:
    """Minibatch gradient and parameter update of the step that produced this state (NaN if none)."""
    nan = np.full(width, np.nan)
    if last is None:
        return {"mb_grad_w_norm": nan, "mb_grad_a": nan, "update_w_norm": nan,
                "update_w_radial": nan, "update_w_tangential": nan, "update_a": nan, "batch_loss": np.nan}
    dW, W_prev = last["dW"].numpy(), last["W_prev"].numpy()
    radial, tangential = geo.radial_tangential(W_prev, dW)
    return {"mb_grad_w_norm": np.linalg.norm(last["grad_W"].numpy(), axis=1),
            "mb_grad_a": last["grad_a"].numpy(), "update_w_norm": np.linalg.norm(dW, axis=1),
            "update_w_radial": radial, "update_w_tangential": tangential,
            "update_a": last["da"].numpy(), "batch_loss": last["loss"]}


def snapshot(net, data, task: int, mu, A, normals, eps) -> dict:
    """Everything measured about the network's current state (full-space quantities)."""
    X_tr, X_te, Y_tr, Y_te = data
    W, b, a, c = params(net)
    gW, gb, ga, gc, tr_loss, tr_acc = full_batch_grads(W, b, a, c, X_tr, Y_tr)
    _, _, _, _, te_loss, te_acc = full_batch_grads(W, b, a, c, X_te, Y_te)
    P_tr, P_te = X_tr @ W.T + b, X_te @ W.T + b
    contrib = torch.relu(P_te) * a                                   # (N, H) output contribution of each unit
    total_std = contrib.sum(1).std().item()
    Wn, bn = W.numpy(), b.numpy()
    norm = np.linalg.norm(Wn, axis=1)
    radial, tangential = geo.radial_tangential(Wn, -gW[task].numpy(), eps)   # of the gradient-descent direction
    return {
        "W": Wn.copy(), "b": bn.copy(), "a": a.numpy().copy(), "c": float(c),
        "norm": norm,
        "angle_2d": geo.signed_angle_2d(Wn, eps) if Wn.shape[1] == 2 else np.full(len(norm), np.nan),
        "angle_to_task_normal": geo.angle_between(Wn, normals[task], eps),
        "cos_to_normals": (Wn @ normals.T) / np.where(norm < eps, np.nan, norm)[:, None],
        "dir_margin": geo.directional_margin(Wn, mu, A, eps),
        "support_max": geo.support_max(Wn, bn, mu, A),
        "support_min": geo.support_min(Wn, bn, mu, A),
        "status": geo.unit_status(Wn, bn, mu, A, eps),
        "act_frac_train": (P_tr > 0).double().mean(0).numpy(),
        "act_frac_test": (P_te > 0).double().mean(0).numpy(),
        "preact_max_train": P_tr.max(0).values.numpy(), "preact_min_train": P_tr.min(0).values.numpy(),
        "preact_max_test": P_te.max(0).values.numpy(), "preact_min_test": P_te.min(0).values.numpy(),
        "contrib_std": contrib.std(0).numpy(),
        "cancellation": contrib.std(0).sum().item() / total_std if total_std > 0 else np.nan,
        "grad_W_current": gW[task].numpy(), "grad_w_norm": gW.norm(dim=2).numpy(),
        "grad_b": gb.numpy(), "grad_a": ga.numpy(), "grad_c": gc.numpy(),
        "grad_w_radial": radial, "grad_w_tangential": tangential,
        "train_loss": tr_loss.numpy(), "train_acc": tr_acc.numpy(),
        "test_loss": te_loss.numpy(), "test_acc": te_acc.numpy(),
    }


# ── One condition ─────────────────────────────────────────────────────────────

def run_condition(cfg: Config, tasks: OrthogonalTasks, condition: str) -> tuple[dict, dict]:
    """Train the continued net (and a fresh net per phase) in one condition; (arrays, summary)."""
    cfg = cfg.resolved()
    mu, A, normals = tasks.condition_mu(condition), tasks.A, tasks.normals
    X_tr = torch.tensor(tasks.inputs("train", condition), dtype=DTYPE)
    X_te = torch.tensor(tasks.inputs("test", condition), dtype=DTYPE)
    Y_tr = torch.tensor(tasks.labels_train, dtype=DTYPE)
    Y_te = torch.tensor(tasks.labels_test, dtype=DTYPE)
    data = (X_tr, X_te, Y_tr, Y_te)
    schedule, S, eps = cfg.schedule, cfg.steps_per_task, cfg.norm_eps

    def snap(net, task):
        return snapshot(net, data, task, mu, A, normals, eps)

    # same init, fresh inits and batch order in both conditions (a live init is checked on the
    # shifted inputs when they are run, so both conditions still start from the same weights)
    live_cond = "shifted" if "shifted" in cfg.conditions else cfg.conditions[0]
    X_live = torch.tensor(tasks.inputs("train", live_cond), dtype=DTYPE)
    net = make_net(cfg, seed_for(cfg.seed, "init"), X_live)
    opt = make_optimizer(cfg, net)
    batches = BatchStream(cfg.n_train, cfg.batch_size, seed_for(cfg.seed, "batches"))
    tracker = UnitTracker(net, X_tr, X_te, mu, A, eps, 0)
    cont, fresh = Log(), Log()
    fresh_trackers, fresh_redraws, last = [], [], None

    for p, task in enumerate(schedule):
        fnet = make_net(cfg, seed_for(cfg.seed, "fresh", p), X_live)
        fresh_redraws.append(fnet.init_redraws)
        fopt = make_optimizer(cfg, fnet)
        ftracker = UnitTracker(fnet, X_tr, X_te, mu, A, eps, p * S)
        flast = None
        for s in range(S):
            t = p * S + s
            if s == 0 or t % cfg.log_every == 0:
                cont.add(t, p, task, snap(net, task), last)
                fresh.add(t, p, task, snap(fnet, task), flast)
            idx = batches.next()                                     # identical minibatch for both nets
            xb, yb = X_tr[idx], Y_tr[task, idx]
            last = train_step(net, opt, xb, yb)
            tracker.update(net, t + 1, last)
            flast = train_step(fnet, fopt, xb, yb)
            ftracker.update(fnet, t + 1, flast)
        fresh.add((p + 1) * S, p, task, snap(fnet, task), flast)
        fresh_trackers.append(ftracker)
    cont.add(len(schedule) * S, len(schedule) - 1, schedule[-1], snap(net, schedule[-1]), last)

    arrays = {**cont.arrays(), **fresh.arrays("fresh_"), **tracker.arrays("")}
    fev = [ft.arrays("") for ft in fresh_trackers]
    arrays["fresh_events"] = np.concatenate(
        [np.column_stack([np.full(len(e["events"]), p), e["events"]]) for p, e in enumerate(fev)]
    ).reshape(-1, 6).astype(np.int64)                                # (phase, step, unit, kind, old, new)
    arrays["fresh_events_info"] = np.concatenate([e["events_info"] for e in fev]).reshape(-1, len(EVENT_INFO))
    arrays["fresh_init_status"] = np.stack([ft.init_status for ft in fresh_trackers])
    arrays["fresh_init_silent_train"] = np.stack([ft.init_silent for ft in fresh_trackers])
    arrays["init_status"] = tracker.init_status
    arrays["init_redraws"] = np.array(net.init_redraws)
    arrays["fresh_init_redraws"] = np.array(fresh_redraws)
    arrays["init_silent_train"] = tracker.init_silent
    arrays["path_length"] = tracker.path_length
    arrays["angle_path"] = tracker.angle_path
    arrays.update(mu=mu, A=A, normals=normals, schedule=np.array(schedule),
                  steps_per_task=np.array(S), norm_eps=np.array(eps))
    return arrays, summarize_condition(cfg, arrays, condition)


# ── Summaries ─────────────────────────────────────────────────────────────────

def deaths(events, info, init_status, init_silent) -> list[dict]:
    """
    The first loss of training signal of every initially active unit (active and not
    silent on the train sample at init): the first step it becomes silent on the
    whole train sample, certified inactive or collapsed, with its mechanism.
    """
    out, seen = [], set()
    for (step, h, kind, old, new), vals in zip(events, info):
        if h in seen or init_status[h] != geo.ACTIVE or init_silent[h]:
            continue
        silenced = kind == EVENT_SILENT and new == 1
        lost = kind == EVENT_STATUS and old == geo.ACTIVE and new != geo.ACTIVE
        if not (silenced or lost):
            continue
        seen.add(h)
        v = dict(zip(EVENT_INFO, vals))
        certified = bool(v["support_max"] <= 0)
        if kind == EVENT_STATUS and new == geo.COLLAPSED or not v["norm"] >= 0 or np.isnan(v["dir_margin"]):
            mech = "norm_collapse"
        elif v["dir_margin"] <= 0:
            mech = "rotation_into_cone"
        elif certified or (v["bias_over_norm"] < 0 and v["train_dir_max"] > 0):
            mech = "bias_driven"
        else:
            mech = "rotation_to_cone_edge"
        out.append({"unit": int(h), "step": int(step), "mechanism": mech, "certified_inactive": certified,
                    **{k: _f(x) for k, x in v.items()}})
    return out


def live(status, act_frac_train) -> int:
    """Units that still get training gradient: not certified inactive or collapsed, and active on some train input."""
    return int(((status == geo.ACTIVE) & (act_frac_train > 0)).sum())


def _f(x):
    """JSON-safe float (None for NaN)."""
    x = float(x)
    return None if np.isnan(x) else x


def steps_to(acc, steps, start, threshold):
    """Steps after `start` until the logged accuracy first reaches the threshold (None if never)."""
    hit = np.flatnonzero(acc >= threshold)
    return int(steps[hit[0]] - start) if len(hit) else None


def summarize_condition(cfg: Config, arr: dict, condition: str) -> dict:
    schedule, S, H = cfg.schedule, cfg.steps_per_task, cfg.width
    steps, phase_of = arr["step"], arr["phase"]
    init_status = arr["init_status"]
    died = deaths(arr["events"], arr["events_info"], init_status, arr["init_silent_train"])
    died_by_unit = {d["unit"]: d for d in died}
    ev = arr["events"]
    flips = ev[ev[:, 2] == EVENT_SIGN]
    revivals = ev[((ev[:, 2] == EVENT_STATUS) & (ev[:, 4] == geo.ACTIVE))
                  | ((ev[:, 2] == EVENT_SILENT) & (ev[:, 4] == 0))]   # back to active, or heard again
    two_d = cfg.input_dim == 2

    units = []
    for h in range(H):
        d = died_by_unit.get(h)
        rec = {
            "unit": h,
            "init_status": geo.STATUS_NAMES[init_status[h]],
            "final_status": geo.STATUS_NAMES[arr["status"][-1, h]],
            "init_silent_on_train_sample": bool(arr["init_silent_train"][h]),
            "final_silent_on_train_sample": bool(arr["act_frac_train"][-1, h] == 0),
            "final_live": bool(arr["status"][-1, h] == geo.ACTIVE and arr["act_frac_train"][-1, h] > 0),
            "act_frac_train_final": _f(arr["act_frac_train"][-1, h]),
            "act_frac_test_final": _f(arr["act_frac_test"][-1, h]),
            "norm_init": _f(arr["norm"][0, h]), "norm_final": _f(arr["norm"][-1, h]),
            "dir_margin_init": _f(arr["dir_margin"][0, h]), "dir_margin_final": _f(arr["dir_margin"][-1, h]),
            "angle_moved_deg": _f(geo.angle_between(arr["W"][0, h], arr["W"][-1, h], cfg.norm_eps)),
            "angle_path_deg": _f(arr["angle_path"][h]), "path_length": _f(arr["path_length"][h]),
            "readout_init": _f(arr["a"][0, h]), "readout_final": _f(arr["a"][-1, h]),
            "readout_sign_flips": int((flips[:, 1] == h).sum()),
            "revived": bool((revivals[:, 1] == h).any()),
            "contrib_std_final": _f(arr["contrib_std"][-1, h]),
            "death": None,
        }
        if two_d:
            rec["angle_2d_init"] = _f(arr["angle_2d"][0, h])
            rec["angle_2d_final"] = _f(arr["angle_2d"][-1, h])
        if d is not None:
            p = min(d["step"] - 1, len(schedule) * S - 1) // S        # the update at step-1 killed it
            rec["death"] = dict(d, phase=int(p), task=int(schedule[p]), steps_after_switch=int(d["step"] - p * S),
                                norm_ratio_to_init=_f(d["norm"] / arr["norm"][0, h]))
        units.append(rec)

    # fresh nets: deaths per phase
    fev, finfo = arr["fresh_events"], arr["fresh_events_info"]
    phases = []
    for p, task in enumerate(schedule):
        c_rows = phase_of == p
        c_end = np.flatnonzero(steps == (p + 1) * S)[-1]
        c_start = np.flatnonzero(steps == p * S)[0]
        f_rows = arr["fresh_phase"] == p
        f_acc = arr["fresh_test_acc"][f_rows, task]
        f_steps = arr["fresh_step"][f_rows]
        f_dead = deaths(fev[fev[:, 0] == p][:, 1:], finfo[fev[:, 0] == p], arr["fresh_init_status"][p],
                        arr["fresh_init_silent_train"][p])
        c_dead = [d for d in died if p * S < d["step"] <= (p + 1) * S]
        acc_rows = np.append(np.flatnonzero(c_rows), c_end)
        phases.append({
            "phase": p, "task": int(task), "start_step": p * S, "end_step": (p + 1) * S,
            "continued_final_test_acc": _f(arr["test_acc"][c_end, task]),
            "fresh_final_test_acc": _f(f_acc[-1]),
            "fresh_minus_continued_acc": _f(f_acc[-1] - arr["test_acc"][c_end, task]),
            "continued_final_test_loss": _f(arr["test_loss"][c_end, task]),
            "fresh_final_test_loss": _f(arr["fresh_test_loss"][f_rows, task][-1]),
            "continued_steps_to_threshold": steps_to(arr["test_acc"][acc_rows, task], steps[acc_rows], p * S,
                                                     cfg.solve_threshold),
            "fresh_steps_to_threshold": steps_to(f_acc, f_steps, p * S, cfg.solve_threshold),
            "fresh_solved": bool(f_acc[-1] >= cfg.solve_threshold),
            "continued_live_start": int(live(arr["status"][c_start], arr["act_frac_train"][c_start])),
            "continued_live_end": int(live(arr["status"][c_end], arr["act_frac_train"][c_end])),
            "fresh_live_start": int(live(arr["fresh_status"][f_rows][0], arr["fresh_act_frac_train"][f_rows][0])),
            "fresh_live_end": int(live(arr["fresh_status"][f_rows][-1], arr["fresh_act_frac_train"][f_rows][-1])),
            "continued_deaths": {m: sum(d["mechanism"] == m for d in c_dead) for m in MECHANISMS},
            "fresh_deaths": {m: sum(d["mechanism"] == m for d in f_dead) for m in MECHANISMS},
        })

    solvable = {}
    for j in range(cfg.n_tasks):
        accs = [ph["fresh_final_test_acc"] for ph in phases if ph["task"] == j]
        solvable[str(j)] = {"min_fresh_final_acc": min(accs), "all_solved": all(a >= cfg.solve_threshold for a in accs)}

    counts = {
        "units": H,
        "initially_inactive": int((init_status == geo.INACTIVE).sum()),
        "initially_collapsed": int((init_status == geo.COLLAPSED).sum()),
        "initially_silent_on_train_sample": int(arr["init_silent_train"].sum()),
        "initially_live": int(((init_status == geo.ACTIVE) & ~arr["init_silent_train"]).sum()),
        "died": len(died),
        **{f"died_{m}": sum(d["mechanism"] == m for d in died) for m in MECHANISMS},
        "died_certified_inactive": sum(d["certified_inactive"] for d in died),
        "died_in_first_phase": sum(d["step"] <= S for d in died),
        "died_after_a_switch": sum(d["step"] > S for d in died),
        "revived": int(len(set(revivals[:, 1].tolist()))),
        "active_at_end": int((arr["status"][-1] == geo.ACTIVE).sum()),
        "live_at_end": live(arr["status"][-1], arr["act_frac_train"][-1]),
        "silent_on_train_sample_at_end_but_not_certified": int(
            ((arr["act_frac_train"][-1] == 0) & (arr["status"][-1] == geo.ACTIVE)).sum()),
        "readout_sign_flips": int(len(flips)),
        "units_with_readout_flip": int(len(set(flips[:, 1].tolist()))),
        "fresh_unit_phases": int(H * len(schedule)),
        "fresh_initially_inactive": int((arr["fresh_init_status"] == geo.INACTIVE).sum()),
        "fresh_initially_silent_on_train_sample": int(arr["fresh_init_silent_train"].sum()),
        "fresh_died": sum(sum(ph["fresh_deaths"].values()) for ph in phases),
        **{f"fresh_died_{m}": sum(ph["fresh_deaths"][m] for ph in phases) for m in MECHANISMS},
        "final_cancellation_index": _f(arr["cancellation"][-1]),
    }
    return {"condition": condition, "mu": arr["mu"].tolist(), "counts": counts,
            "fresh_solvable": solvable, "all_tasks_solvable_by_fresh": all(v["all_solved"] for v in solvable.values()),
            "units": units, "phases": phases}


def aggregate(run_dir: Path) -> dict:
    """Sum the counts of every seed in a run folder, per condition, and pool the phase gaps."""
    out = {}
    seeds = sorted(run_dir.glob("seed*/summary.json"))
    for path in seeds:
        summ = json.loads(path.read_text())
        for cond, s in summ["conditions"].items():
            agg = out.setdefault(cond, {"seeds": [], "counts": {}, "gaps": [], "solved": [], "fresh_min_acc": {},
                                        "seeds_all_tasks_solvable": 0, "death_tasks": {}, "deaths": [],
                                        "hazard": {}})
            agg["seeds"].append(summ["seed"])
            for k, v in s["counts"].items():
                if isinstance(v, (int, float)) and k != "final_cancellation_index":
                    agg["counts"][k] = agg["counts"].get(k, 0) + v
            agg["seeds_all_tasks_solvable"] += int(s["all_tasks_solvable_by_fresh"])
            for j, v in s["fresh_solvable"].items():
                agg["fresh_min_acc"][j] = min(agg["fresh_min_acc"].get(j, 1.0), v["min_fresh_final_acc"])
            agg["gaps"].append([ph["fresh_minus_continued_acc"] for ph in s["phases"]])
            agg["solved"].append([ph["fresh_solved"] for ph in s["phases"]])
            for ph in s["phases"]:
                # deaths per live unit at the phase start, continued vs fresh (same task, same batches)
                hz = agg["hazard"].setdefault(str(ph["phase"]), {"task": ph["task"], "continued_live": 0,
                                                                 "continued_died": 0, "fresh_live": 0, "fresh_died": 0})
                hz["continued_live"] += ph["continued_live_start"]
                hz["continued_died"] += sum(ph["continued_deaths"].values())
                hz["fresh_live"] += ph["fresh_live_start"]
                hz["fresh_died"] += sum(ph["fresh_deaths"].values())
            for u in s["units"]:
                if u["death"]:
                    t = str(u["death"]["task"])
                    agg["death_tasks"][t] = agg["death_tasks"].get(t, 0) + 1
                    agg["deaths"].append({"seed": summ["seed"], **u["death"]})
    for agg in out.values():
        g = np.array(agg.pop("gaps"), dtype=float)
        solved = np.array(agg.pop("solved"), dtype=bool)
        g_solved = np.where(solved, g, np.nan)
        with np.errstate(all="ignore"):
            agg["fresh_minus_continued_acc_per_phase"] = {
                "mean": np.nanmean(g, 0).tolist(), "min": np.nanmin(g, 0).tolist(), "max": np.nanmax(g, 0).tolist(),
                "mean_where_fresh_solved": [None if np.isnan(v) else float(v) for v in np.nanmean(g_solved, 0)],
                "n_seeds_fresh_solved": solved.sum(0).tolist()}
    (run_dir / "aggregate.json").write_text(json.dumps(out, indent=2))
    return out


# ── Seed driver, report ───────────────────────────────────────────────────────

def run_seed(cfg: Config, out_dir: Path | None = None) -> dict:
    """Both conditions of one seed; saved under out_dir/seed<s>/ when out_dir is given."""
    cfg = cfg.resolved()
    tasks = build_tasks(cfg.n_tasks, cfg.input_dim, cfg.n_train, cfg.n_test, seed_for(cfg.seed, "data"),
                        mu=cfg.mu, scales=cfg.scales, normals=cfg.task_normals, label_rule=cfg.label_rule)
    summary = {"seed": cfg.seed, "config": asdict(cfg), "conditions": {}}
    results = {}
    for cond in cfg.conditions:
        arrays, summ = run_condition(cfg, tasks, cond)
        results[cond] = arrays
        summary["conditions"][cond] = summ
    if out_dir is not None:
        sd = out_dir / f"seed{cfg.seed}"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        for cond, arrays in results.items():
            np.savez_compressed(sd / f"{cond}.npz", **arrays)
        (sd / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "arrays": results}


def num(x, spec: str) -> str:
    """Format a number, or 'n/a' for a missing one (e.g. no direction for a collapsed unit)."""
    return "n/a" if x is None else format(x, spec)


def print_seed_report(summary: dict) -> None:
    for cond, s in summary["conditions"].items():
        c = s["counts"]
        solv = ", ".join(f"task {j}: {v['min_fresh_final_acc']:.3f}" for j, v in s["fresh_solvable"].items())
        print(f"  [{cond:8s}] fresh min acc {solv} -> {'solvable' if s['all_tasks_solvable_by_fresh'] else 'NOT solvable'}")
        print(f"             units {c['units']}: init certified inactive {c['initially_inactive']}, "
              f"init live {c['initially_live']}; died {c['died']} = {died_text(c)} "
              f"(certified {c['died_certified_inactive']}; first phase {c['died_in_first_phase']}, "
              f"after a switch {c['died_after_a_switch']}); live at end {c['live_at_end']}; "
              f"readout flips {c['readout_sign_flips']}")
        print(f"             fresh nets ({c['fresh_unit_phases']} unit-phases): init certified inactive "
              f"{c['fresh_initially_inactive']}; died {c['fresh_died']} = {died_text(c, 'fresh_')}")
        gaps = " ".join(f"{ph['fresh_minus_continued_acc']:+.3f}" for ph in s["phases"])
        print(f"             fresh − continued final test acc per phase: {gaps}")
        for u in s["units"]:
            d = u["death"]
            how = (f"died step {d['step']} (phase {d['phase']}, task {d['task']}, {d['mechanism']}, "
                   f"dir margin {num(d['dir_margin'], '+.3f')}, norm x{num(d['norm_ratio_to_init'], '.3f')}, "
                   f"test act {num(d['test_act_frac'], '.4f')})") if d else ""
            ang = (f"angle {u['angle_2d_init']:7.1f} -> {u['angle_2d_final']:7.1f}  "
                   if "angle_2d_init" in u and u["angle_2d_init"] is not None and u["angle_2d_final"] is not None else "")
            print(f"             unit {u['unit']}: {u['init_status']:>18s} -> {u['final_status']:<18s} {ang}"
                  f"|w| {u['norm_init']:.2f} -> {u['norm_final']:.2f}  a {u['readout_init']:+.2f} -> "
                  f"{u['readout_final']:+.2f} flips {u['readout_sign_flips']}  {how}")


def task_name_of(j: int) -> str:
    return chr(ord("A") + j) if j < 26 else str(j)


def died_text(c: dict, prefix: str = "") -> str:
    return ", ".join(f"{m} {c[f'{prefix}died_{m}']}" for m in MECHANISMS)


def print_aggregate(agg: dict) -> None:
    print("\nAggregate over all seeds in the folder:")
    for cond, a in agg.items():
        c = a["counts"]
        print(f"  [{cond:8s}] seeds {a['seeds']}  fresh-solvable seeds {a['seeds_all_tasks_solvable']}/{len(a['seeds'])}  "
              f"fresh min acc per task {a['fresh_min_acc']}")
        print(f"             units {c['units']}: init certified inactive {c['initially_inactive']}, init live "
              f"{c['initially_live']}; died {c['died']} = {died_text(c)} (certified {c['died_certified_inactive']}, "
              f"after a switch {c['died_after_a_switch']}); revived {c['revived']}; live at end {c['live_at_end']}")
        print(f"             fresh unit-phases {c['fresh_unit_phases']}: init certified inactive "
              f"{c['fresh_initially_inactive']}; died {c['fresh_died']} = {died_text(c, 'fresh_')}")
        gp = a["fresh_minus_continued_acc_per_phase"]
        print("             mean fresh − continued acc per phase: " + " ".join(f"{v:+.3f}" for v in gp["mean"]))
        print("             same, phases where the fresh net solved the task only: " + " ".join(
            "  n/a " if v is None else f"{v:+.3f}" for v in gp["mean_where_fresh_solved"])
              + f"  (seeds per phase {gp['n_seeds_fresh_solved']})")
        print("             deaths / live units at phase start, continued | fresh: " + "  ".join(
            f"{p}:{task_name_of(h['task'])} {h['continued_died']}/{h['continued_live']} | "
            f"{h['fresh_died']}/{h['fresh_live']}" for p, h in a["hazard"].items()))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    p.add_argument("--n-tasks", "--n_tasks", type=int, default=d.n_tasks)
    p.add_argument("--input-dim", "--input_dim", type=int, default=None, help="D (default max(2, K))")
    p.add_argument("--mu", type=float, nargs="+", default=None, help="D values (default e_{D-1})")
    p.add_argument("--scales", type=float, nargs="+", default=None, help="diagonal of A, D values")
    p.add_argument("--width", type=int, default=d.width)
    p.add_argument("--hidden-bias", "--hidden_bias", choices=["zero", "trainable"], default=d.hidden_bias)
    p.add_argument("--init", choices=["kaiming", "live"], default=d.init,
                   help="live: redraw hidden units that start silent on the shifted train inputs")
    p.add_argument("--bias-init", "--bias_init", choices=["zero", "uniform"], default=d.bias_init,
                   help="hidden bias at init (with --hidden-bias trainable): zero, or uniform U(-1/sqrt(D), 1/sqrt(D))")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default=d.optimizer)
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--batch-size", "--batch_size", type=int, default=d.batch_size)
    p.add_argument("--steps-per-task", "--steps_per_task", type=int, default=d.steps_per_task)
    p.add_argument("--cycles", type=int, default=d.cycles, help="passes through all K tasks")
    p.add_argument("--label-rule", "--label_rule", choices=list(LABEL_RULES), default=d.label_rule,
                   help="halfspace: y = 1[v·(x−μ) > 0]; stripe: y = 1[|v·(x−μ)| < h], a band that one unit can't fit; "
                        "xor (K = 2): A = quadrant XOR, B = the same turned 45°, 4 units each")
    p.add_argument("--task-normals", "--task_normals", choices=list(NORMAL_KINDS), default=d.task_normals,
                   help="task hyperplane normals: coordinate axes, (e_0 ± e_1)/√2 for K = 2, or those plus "
                        "C = (e_0 + e_2)/√2 for K = 3 (diagonal-c, not orthogonal)")
    p.add_argument("--phases", type=int, default=None,
                   help="total number of phases (switches = phases - 1); overrides --cycles")
    p.add_argument("--n-train", "--n_train", type=int, default=d.n_train)
    p.add_argument("--n-test", "--n_test", type=int, default=d.n_test)
    p.add_argument("--log-every", "--log_every", type=int, default=d.log_every)
    p.add_argument("--norm-eps", "--norm_eps", type=float, default=d.norm_eps)
    p.add_argument("--solve-threshold", "--solve_threshold", type=float, default=d.solve_threshold)
    p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", type=Path, default=RESULTS, help="results root")
    p.add_argument("--tag", default="", help="suffix of the run folder name")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fields = {k: v for k, v in vars(args).items() if k not in ("seeds", "out", "tag")}
    base = Config(**fields).resolved()
    run_dir = args.out / (run_name(base) + (f"_{args.tag}" if args.tag else ""))
    print(f"Orthogonal tasks: K={base.n_tasks} D={base.input_dim} width={base.width} mu={base.mu} "
          f"scales={base.scales} schedule={base.schedule} -> {run_dir}")
    for seed in args.seeds:
        t0 = time.time()
        res = run_seed(Config(**{**asdict(base), "seed": seed}), run_dir)
        print(f"seed {seed} ({time.time() - t0:.1f}s)")
        print_seed_report(res["summary"])
    print_aggregate(aggregate(run_dir))
    print(f"\nPlots: uv run python -m visualization_scripts.toy_models.plot_orthogonal_tasks --run-dir {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
