"""The run configuration, its human-readable folder name, and the reproducible seed streams."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from orthogonal_tasks.data import LABEL_RULES, NORMAL_KINDS, default_scales, resolve_input_dim, task_schedule

BIAS_INITS = ("uniform", "zero")


@dataclass
class Config:
    n_tasks: int = 2
    input_dim: int | None = None          # None: max(2, n_tasks)
    scales: list[float] | None = None     # None: 1 everywhere, 0.5 on the last axis
    label_rule: str = "halfspace"         # see data.py
    task_normals: str = "axes"            # see data.py
    width: int = 4
    bias_init: str = "uniform"            # hidden bias at init: U(-1/sqrt(D), 1/sqrt(D)) or zero
    optimizer: str = "sgd"
    momentum: float = 0.0
    lr: float = 0.1
    batch_size: int = 64
    steps_per_task: int = 10_000
    phases: int = 11                      # phases in total, so phases - 1 switches
    n_train: int = 2048
    n_test: int = 2048
    log_every: int = 10
    solve_threshold: float = 0.95         # a fresh net "solves" a task at this final test accuracy
    seed: int = 0

    def resolved(self) -> "Config":
        """A copy with D and the scales filled in and every field validated."""
        d = resolve_input_dim(self.n_tasks, self.input_dim)
        scales = default_scales(d) if self.scales is None else np.asarray(self.scales, float)
        if scales.shape != (d,) or np.any(scales <= 0):
            raise ValueError(f"--scales needs D={d} positive values")
        if self.label_rule not in LABEL_RULES:
            raise ValueError(f"--label-rule is one of {LABEL_RULES}")
        if self.task_normals not in NORMAL_KINDS:
            raise ValueError(f"--task-normals is one of {NORMAL_KINDS}")
        if self.label_rule == "xor" and (self.n_tasks != 2 or self.task_normals != "axes"):
            raise ValueError("--label-rule xor needs --n-tasks 2 and --task-normals axes")
        if self.task_normals == "diagonal" and self.n_tasks != 2:
            raise ValueError("--task-normals diagonal needs --n-tasks 2")
        if self.task_normals == "diagonal-c" and (self.n_tasks != 3 or d < 3):
            raise ValueError("--task-normals diagonal-c needs --n-tasks 3 and D >= 3")
        if self.bias_init not in BIAS_INITS:
            raise ValueError(f"--bias-init is one of {BIAS_INITS}")
        if min(self.width, self.steps_per_task, self.phases, self.log_every, self.batch_size) < 1:
            raise ValueError("width, steps per task, phases, log_every and batch size must be >= 1")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError("--optimizer is 'sgd' or 'adam'")
        out = Config(**asdict(self))
        out.input_dim, out.scales = d, scales.tolist()
        return out

    @property
    def schedule(self) -> list[int]:
        return task_schedule(self.n_tasks, self.phases)


def run_name(cfg: Config) -> str:
    """Folder name of a configuration (seed excluded), e.g. 2d_AB-axes-stripes_thin-x2_4-units_10k-steps_10-switches.
    Defaults of the bias init, optimiser, learning rate, batch size and sample size are left out; other
    values are appended."""
    cfg = cfg.resolved()
    k = cfg.n_tasks
    tasks = "".join(chr(ord("A") + j) for j in range(k)) if k <= 26 else f"{k}tasks"
    normals = "diagonal" if cfg.task_normals == "diagonal-c" else cfg.task_normals   # ABC already names task C
    rule = {"stripe": "-stripes", "xor": "-xor"}.get(cfg.label_rule, "")
    scales = np.asarray(cfg.scales)
    off = np.flatnonzero(~np.isclose(scales, 1))
    if len(off) == 0:
        cloud = "round"
    elif len(off) == 1 and np.isclose(scales[off[0]], 0.5):
        cloud = f"thin-x{off[0] + 1}"
    else:
        cloud = "scales-" + ",".join(f"{x:g}" for x in scales)
    S = cfg.steps_per_task
    steps = f"{S // 1000}k-steps" if S % 1000 == 0 else f"{S}-steps"
    parts = [f"{cfg.input_dim}d", f"{tasks}-{normals}{rule}", cloud, f"{cfg.width}-units", steps,
             f"{cfg.phases - 1}-switches"]
    d0 = Config()
    if cfg.bias_init != d0.bias_init:
        parts.append(f"{cfg.bias_init}-bias-init")
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


def task_name(j: int) -> str:
    return chr(ord("A") + j) if j < 26 else str(j)
