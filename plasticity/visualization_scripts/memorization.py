"""
visualization_scripts/memorization.py — Plots for the memorization plasticity setting.

Usage:
    python -m visualization_scripts.memorization results/memorization/run_name.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from visualization_scripts.common import (
    set_style, plot_learning_curves, plot_bar, smooth, steps_and_values,
    add_task_switch_vlines, PALETTE,
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "memorization"


# ── Individual plot functions ──────────────────────────────────────────────────

def plot_overwrite_retention(
    log:        list[dict],
    save_path:  Optional[str] = None,
    smooth_win: int = 3,
) -> plt.Figure:
    """
    Two-panel figure:
        Top:    current-label accuracy on changed vs unchanged subset
        Bottom: previous-label prediction rate (forgetting old associations)
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    steps_c, acc_changed  = steps_and_values(log, "current_label_acc_changed")
    steps_u, retention    = steps_and_values(log, "unchanged_retention")
    steps_p, prev_rate    = steps_and_values(log, "previous_label_pred_rate")

    # detect change events from log
    event_steps = [e["step"] for e in log if e.get("n_change_events", 0) > (
        log[max(0, i-1)].get("n_change_events", 0)
        if i > 0 else 0
    ) for i, e in enumerate(log)]

    ax1.plot(steps_c, smooth(acc_changed, smooth_win),
             color=PALETTE[0], label="changed subset (current label)")
    ax1.plot(steps_u, smooth(retention,   smooth_win),
             color=PALETTE[1], label="unchanged subset")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.set_title("Memorization: accuracy by subset")

    ax2.plot(steps_p, smooth(prev_rate, smooth_win),
             color=PALETTE[2], label="previous-label prediction rate")
    ax2.set_ylabel("Rate")
    ax2.set_xlabel("Training step")
    ax2.set_ylim(0, 1.05)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_overall_accuracy(
    log:        list[dict],
    save_path:  Optional[str] = None,
    smooth_win: int = 3,
) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(8, 3))
    steps, acc = steps_and_values(log, "overall_accuracy")
    ax.plot(steps, smooth(acc, smooth_win), color=PALETTE[0])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Overall accuracy")
    ax.set_title("Overall current-label accuracy")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_persistence_histogram(
    log:       list[dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Distribution of mean association persistence over the run."""
    set_style()
    _, vals = steps_and_values(log, "mean_persistence")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(_, vals, color=PALETTE[3])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean persistence (steps)")
    ax.set_title("Association persistence over time")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def compare_configs(
    result_files: list[str],
    metric:       str = "current_label_acc_changed",
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Overlay learning curves from multiple result JSON files.
    """
    logs = {}
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        label = data.get("run_name", Path(path).stem)
        logs[label] = data["log"]

    return plot_learning_curves(
        logs, metric=metric,
        title=f"Memorization — {metric}",
        ylabel=metric,
        save_path=save_path,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("result_file",  help="Path to a memorization result JSON")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    with open(args.result_file) as f:
        data = json.load(f)

    run_name = data.get("run_name", Path(args.result_file).stem)
    out_dir  = Path(args.output_dir) if args.output_dir else Path(args.result_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log = data["log"]
    plot_overwrite_retention(log,   save_path=str(out_dir / f"{run_name}_overwrite_retention.png"))
    plot_overall_accuracy(log,       save_path=str(out_dir / f"{run_name}_overall_acc.png"))
    plot_persistence_histogram(log,  save_path=str(out_dir / f"{run_name}_persistence.png"))
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
