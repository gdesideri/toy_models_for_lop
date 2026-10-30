"""
One seed: the continued net trains through every phase (net and optimizer state carry across
switches); at every phase a fresh net, newly initialised, trains next to it on the identical minibatches.

Logged every `log_every` steps, at every switch and at the end, for both (the fresh ones with the
prefix fresh_ and the phase in fresh_phase):
    W, b, a, c              hidden weights, hidden biases, readout weights, output bias
    act_frac_train          fraction of train inputs on which each unit is active (0 = silent everywhere)
    preact_max_train        max over train inputs of w·x + b (how far a unit is from going silent)
    grad_b                  (K, H) full-batch gradient of each task's loss on each bias (the push on it)
    train/test loss, acc    (K,) of every task
Death of a unit: the first step after which it is silent on every train input. It then gets no gradient
on w or b, so under plain SGD it stays silent for good. Checked after every step, not only when logged.
"""
from __future__ import annotations

import numpy as np
import torch

from orthogonal_tasks.config import Config, seed_for
from orthogonal_tasks.data import Tasks, build_tasks
from orthogonal_tasks.network import (DTYPE, BatchStream, full_batch_grads, make_net, make_optimizer, params,
                                      train_step)


def build_seed_tasks(cfg: Config) -> Tasks:
    """The data of a seed (the same for every run of that seed with the same task settings)."""
    cfg = cfg.resolved()
    return build_tasks(cfg.n_tasks, cfg.input_dim, cfg.n_train, cfg.n_test, seed_for(cfg.seed, "data"),
                       scales=cfg.scales, normals=cfg.task_normals, label_rule=cfg.label_rule)


def snapshot(net, X_tr, X_te, Y_tr, Y_te) -> dict:
    """Everything logged about a net's current state."""
    W, b, a, c = params(net)
    _, gb, _, _, tr_loss, tr_acc = full_batch_grads(W, b, a, c, X_tr, Y_tr)
    _, _, _, _, te_loss, te_acc = full_batch_grads(W, b, a, c, X_te, Y_te)
    P = X_tr @ W.T + b
    return {"W": W.numpy().copy(), "b": b.numpy().copy(), "a": a.numpy().copy(), "c": float(c),
            "act_frac_train": (P > 0).double().mean(0).numpy(), "preact_max_train": P.max(0).values.numpy(),
            "grad_b": gb.numpy(), "train_loss": tr_loss.numpy(), "train_acc": tr_acc.numpy(),
            "test_loss": te_loss.numpy(), "test_acc": te_acc.numpy()}


class Log:
    """Logged snapshots of one net (or of the fresh nets, one after the other)."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, step: int, phase: int, task: int, snap: dict) -> None:
        self.rows.append(dict(snap, step=step, phase=phase, task=task))

    def arrays(self, prefix: str = "") -> dict:
        return {prefix + k: np.stack([np.asarray(r[k]) for r in self.rows]) for k in self.rows[0]}


class DeathWatch:
    """The first step after which each unit is silent on every train input (-1 while alive)."""

    def __init__(self, width: int):
        self.step = np.full(width, -1, dtype=np.int64)

    def check(self, net, X_tr, step: int) -> None:
        W, b, _, _ = params(net)
        silent = ((X_tr @ W.T + b).max(0).values <= 0).numpy()
        self.step[silent & (self.step < 0)] = step


def run_seed(cfg: Config) -> dict:
    """Train one seed; returns the logged arrays (continued net, then fresh_ for the fresh nets)."""
    cfg = cfg.resolved()
    tasks = build_seed_tasks(cfg)
    X_tr = torch.tensor(tasks.inputs("train"), dtype=DTYPE)
    X_te = torch.tensor(tasks.inputs("test"), dtype=DTYPE)
    Y_tr = torch.tensor(tasks.labels_train, dtype=DTYPE)
    Y_te = torch.tensor(tasks.labels_test, dtype=DTYPE)
    schedule, S = cfg.schedule, cfg.steps_per_task

    def snap(net):
        return snapshot(net, X_tr, X_te, Y_tr, Y_te)

    net = make_net(cfg, seed_for(cfg.seed, "init"), X_tr)
    opt = make_optimizer(cfg, net)
    batches = BatchStream(cfg.n_train, cfg.batch_size, seed_for(cfg.seed, "batches"))
    cont, fresh, watch = Log(), Log(), DeathWatch(cfg.width)
    fresh_deaths, fresh_redraws = [], []

    for p, task in enumerate(schedule):
        fnet = make_net(cfg, seed_for(cfg.seed, "fresh", p), X_tr)
        fopt = make_optimizer(cfg, fnet)
        fwatch = DeathWatch(cfg.width)
        fresh_redraws.append(fnet.init_redraws)
        for s in range(S):
            t = p * S + s
            if s == 0 or t % cfg.log_every == 0:
                cont.add(t, p, task, snap(net))
                fresh.add(t, p, task, snap(fnet))
            idx = batches.next()                                     # the same minibatch for both nets
            xb, yb = X_tr[idx], Y_tr[task, idx]
            train_step(net, opt, xb, yb)
            watch.check(net, X_tr, t + 1)
            train_step(fnet, fopt, xb, yb)
            fwatch.check(fnet, X_tr, t + 1)
        fresh.add((p + 1) * S, p, task, snap(fnet))
        fresh_deaths.append(fwatch.step)
    cont.add(len(schedule) * S, len(schedule) - 1, schedule[-1], snap(net))

    return {**cont.arrays(), **fresh.arrays("fresh_"),
            "death_step": watch.step, "fresh_death_step": np.stack(fresh_deaths),
            "init_redraws": np.array(net.init_redraws), "fresh_init_redraws": np.array(fresh_redraws),
            "A": tasks.A, "normals": tasks.normals, "schedule": np.array(schedule), "steps_per_task": np.array(S)}
