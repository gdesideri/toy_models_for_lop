"""
visualization_scripts/representation.py — Plots for the representation plasticity setting.

Usage:
    python -m visualization_scripts.representation results/representation/run_name.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from visualization_scripts.common import (
    set_style, plot_learning_curves, plot_heatmap, plot_bar,
    smooth, steps_and_values, add_task_switch_vlines, PALETTE,
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "representation"


# ── Relevance heatmap ─────────────────────────────────────────────────────────

def plot_relevance_heatmap(
    relevance_matrix: np.ndarray,
    task_names:       list[str],
    factor_names:     list[str],
    save_path:        Optional[str] = None,
) -> plt.Figure:
    """
    Task-by-feature relevance heatmap (R_{t,j}).
    Rows = tasks, columns = latent factors.
    """
    short_tasks = [t.split("_", 1)[-1] for t in task_names]
    return plot_heatmap(
        relevance_matrix,
        row_labels=short_tasks,
        col_labels=factor_names,
        title="Interventional relevance R_{t,j}",
        save_path=save_path,
    )


# ── Task similarity matrices ──────────────────────────────────────────────────

def plot_similarity_matrices(
    feat_sim:   np.ndarray,
    label_sim:  np.ndarray,
    task_names: list[str],
    save_path:  Optional[str] = None,
) -> plt.Figure:
    set_style()
    short = [t.split("_", 1)[-1] for t in task_names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, mat, title in [
        (ax1, feat_sim,  "Feature similarity (Jaccard)"),
        (ax2, label_sim, "Label correlation"),
    ]:
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right")
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short)
        ax.set_title(title)
        for i in range(len(short)):
            for j in range(len(short)):
                ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Learning curve with task timeline ────────────────────────────────────────

def plot_accuracy_timeline(
    log:           list[dict],
    task_names:    list[str],
    switch_every:  int,
    save_path:     Optional[str] = None,
    smooth_win:    int = 5,
) -> plt.Figure:
    """
    Accuracy over time with task-switch markers and task-name annotations.
    """
    set_style()
    steps, accs = steps_and_values(log, "current_acc")
    if len(steps) == 0:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, smooth(accs, smooth_win), color=PALETTE[0])

    n_tasks    = len(task_names)
    max_step   = int(steps[-1])
    switches   = list(range(switch_every, max_step, switch_every))
    add_task_switch_vlines(ax, switches)

    # annotate task blocks
    for k, tname in enumerate(task_names):
        mid = (k + 0.5) * switch_every
        if mid <= max_step:
            short = tname.split("_", 1)[-1]
            ax.text(mid, 0.02, short, ha="center", va="bottom",
                    fontsize=7, color="grey", rotation=45)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy (current task)")
    ax.set_title("Representation plasticity — accuracy timeline")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Per-task bar charts ───────────────────────────────────────────────────────

def plot_per_task_metrics(
    per_task:  dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    set_style()
    task_names = list(per_task.keys())
    aucs       = [per_task[t].get("auc",           0.0) or 0.0 for t in task_names]
    final_accs = [per_task[t].get("final_acc",     0.0) or 0.0 for t in task_names]
    adapt_raw  = [per_task[t].get("adaptation_steps") for t in task_names]
    adapt_max  = max((a for a in adapt_raw if a is not None), default=1) or 1
    adapt_norm = [(a / adapt_max if a is not None else 1.0) for a in adapt_raw]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    short = [t.split("_", 1)[-1] for t in task_names]

    for ax, vals, title, ylabel in [
        (axes[0], final_accs,  "Final accuracy",         "Accuracy"),
        (axes[1], aucs,        "Learning curve AUC",     "AUC"),
        (axes[2], adapt_norm,  "Adaptation speed (norm)", "Frac. of max steps"),
    ]:
        ax.bar(range(len(short)), vals, color=PALETTE[:len(short)])
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Feature-relevance timeline ────────────────────────────────────────────────

def plot_feature_relevance_timeline(
    relevance_matrix: np.ndarray,
    task_names:       list[str],
    factor_names:     list[str],
    switch_every:     int,
    save_path:        Optional[str] = None,
) -> plt.Figure:
    """
    Line plot where each line is a factor, x-axis indexes tasks (time).
    Useful for FeatureIntroductionStream.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    n_tasks, n_factors = relevance_matrix.shape
    x = np.arange(n_tasks) * switch_every

    for j in range(n_factors):
        ax.plot(x, relevance_matrix[:, j],
                marker="o", label=factor_names[j], color=PALETTE[j % len(PALETTE)])

    ax.set_xlabel("Training step (task midpoint)")
    ax.set_ylabel("Interventional relevance R_{t,j}")
    ax.set_title("Feature relevance over task sequence")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("result_file",  help="Path to a representation result JSON")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    with open(args.result_file) as f:
        data = json.load(f)

    run_name    = data.get("run_name", Path(args.result_file).stem)
    out_dir     = Path(args.output_dir) if args.output_dir else Path(args.result_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names   = data["tasks"]
    factor_names = data["factor_names"]
    switch_every = data["config"]["switch_every"]
    log          = data["log"]
    per_task     = data["per_task"]
    relevance    = np.array(data["relevance_matrix"])
    feat_sim     = np.array(data["feat_sim"])
    label_sim    = np.array(data["label_sim"])

    plot_relevance_heatmap(relevance, task_names, factor_names,
        save_path=str(out_dir / f"{run_name}_relevance.png"))
    plot_similarity_matrices(feat_sim, label_sim, task_names,
        save_path=str(out_dir / f"{run_name}_similarity.png"))
    plot_accuracy_timeline(log, task_names, switch_every,
        save_path=str(out_dir / f"{run_name}_timeline.png"))
    plot_per_task_metrics(per_task,
        save_path=str(out_dir / f"{run_name}_per_task.png"))
    plot_feature_relevance_timeline(relevance, task_names, factor_names, switch_every,
        save_path=str(out_dir / f"{run_name}_feature_timeline.png"))

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
