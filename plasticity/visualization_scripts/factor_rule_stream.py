"""
visualization_scripts/factor_rule_stream.py — Figures for experiments/run_factor_rule_stream.py.

    probe_accuracy.png          linear-probe test accuracy of every factor on the frozen z
    current_task_accuracy.png   test accuracy on the current rule, against task and examples seen
    all_rule_evaluation.png     the head after each task under each of the six rules
    within_task_adaptation.png  train accuracy over the steps of a task, early vs late tasks

Reads the CSVs of a run directory, so it can be re-run on its own:
    uv run python -m visualization_scripts.factor_rule_stream <run_dir>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: figures are written to disk
import matplotlib.pyplot as plt

from visualization_scripts import common
from tasks.factor_rules import FACTORS


def _read_csv(path):
    """Columns of a CSV as numpy arrays (numeric where possible)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            cols[k] = np.array(vals, dtype=float)
        except ValueError:
            cols[k] = np.array(vals)
    return cols


def plot_probe_accuracy(run_dir):
    probes = _read_csv(run_dir / "representation_probe_results.csv")
    fig = common.plot_bar(dict(zip(probes["factor"], probes["test_accuracy"])),
                          title="Linear probes on the frozen representation",
                          ylabel="test accuracy")
    ax = fig.axes[0]
    ax.set_ylim(0, 1.08)
    # chance level of each factor, 1 / K
    for i, k in enumerate(probes["classes"]):
        ax.hlines(1 / k, i - 0.4, i + 0.4, color="0.2", linestyle="--", linewidth=1.0,
                  label="chance (1/K)" if i == 0 else None)
    ax.legend(loc="upper right")
    common.save(fig, run_dir / "probe_accuracy.png")


def plot_current_task_accuracy(run_dir):
    """Current-rule test accuracy against task id (left) and total examples seen (right)."""
    common.set_style()
    tasks = _read_csv(run_dir / "task_end_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, x, xlabel in ((axes[0], tasks["task_id"], "task"),
                          (axes[1], tasks["total_examples_seen"], "total examples seen")):
        for i, name in enumerate(FACTORS):
            m = tasks["relevant_factor"] == name
            ax.scatter(x[m], tasks["current_task_test_accuracy"][m], s=9, zorder=2,
                       color=common.PALETTE[i % len(common.PALETTE)], label=name)
        ax.axhline(0.5, color="0.4", linestyle=":", linewidth=0.8)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("test accuracy on the current rule")
    axes[0].set_ylim(0.4, 1.02)
    axes[1].legend(title="current rule", loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=1.5)
    common.bold_suptitle(fig, "Persistent head: accuracy at the end of each task")
    fig.tight_layout()
    common.save(fig, run_dir / "current_task_accuracy.png")


def plot_all_rule_evaluation(run_dir):
    """One panel per evaluated rule: its test accuracy after every task, marked when it was the trained rule."""
    common.set_style()
    ev = _read_csv(run_dir / "all_rule_evaluation.csv")
    fig, axes = plt.subplots(len(FACTORS), 1, figsize=(14, 2.0 * len(FACTORS)), sharex=True, sharey=True)
    for i, (ax, name) in enumerate(zip(axes, FACTORS)):
        m = ev["evaluated_factor"] == name
        t, acc, cur = ev["after_task"][m], ev["test_accuracy"][m], ev["is_current_rule"][m] == 1
        color = common.PALETTE[i % len(common.PALETTE)]
        ax.plot(t, acc, color=color, linewidth=0.7, alpha=0.6, label="after any task")
        ax.scatter(t[cur], acc[cur], color=color, s=10, zorder=3, label=f"after a {name} task")
        ax.axhline(0.5, color="0.4", linestyle=":", linewidth=0.8)
        ax.set_ylabel("test acc")
        ax.set_title(f"rule: {name}", loc="left")
        ax.legend(loc="lower right", fontsize="small", ncol=2)
    axes[-1].set_xlabel("after task")
    axes[0].set_ylim(0.3, 1.02)
    common.bold_suptitle(fig, "The same head evaluated under all six rules")
    fig.tight_layout()
    common.save(fig, run_dir / "all_rule_evaluation.png")


def plot_within_task_adaptation(run_dir, n_early=60, n_late=60):
    """
    Mean train accuracy at each step of a task, per rule: over the first `n_early` tasks
    (left) and the last `n_late` (right). Step 1 is the first batch after the switch.
    """
    common.set_style()
    steps = _read_csv(run_dir / "continual_training.csv")
    last = steps["task_id"].max()
    windows = (("first", steps["task_id"] < n_early), ("last", steps["task_id"] > last - n_late))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, (which, w) in zip(axes, windows):
        n_tasks = len(np.unique(steps["task_id"][w]))
        for i, name in enumerate(FACTORS):
            m = w & (steps["relevant_factor"] == name)
            s = steps["step_within_task"][m].astype(int)
            mean = np.bincount(s, weights=steps["train_accuracy"][m]) / np.maximum(np.bincount(s), 1)
            ax.plot(np.arange(1, len(mean)), mean[1:], label=name, **common.style_for(i))
        ax.axhline(0.5, color="0.4", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("step within task")
        ax.set_title(f"{which} {n_tasks} tasks")
    axes[0].set_ylabel("train accuracy (batch, mean over tasks)")
    axes[1].legend(title="current rule", loc="upper left", bbox_to_anchor=(1.02, 1))
    common.bold_suptitle(fig, "Adaptation after a rule switch")
    fig.tight_layout()
    common.save(fig, run_dir / "within_task_adaptation.png")


def plot_all(run_dir):
    run_dir = Path(run_dir)
    plot_probe_accuracy(run_dir)
    plot_current_task_accuracy(run_dir)
    plot_all_rule_evaluation(run_dir)
    plot_within_task_adaptation(run_dir)


if __name__ == "__main__":
    plot_all(sys.argv[1])
