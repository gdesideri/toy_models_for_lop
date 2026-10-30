"""
plot_orthogonal_summary.py — One compact figure over several runs of experiments/run_orthogonal_tasks.py:
live hidden units and held-out accuracy over training, continued net vs fresh nets, every seed.

One row per run, two columns per condition present (live units, accuracy): shifted first, then centered.
    live units   units active on the train sample (status active and firing on >= 1 train input); seeds are
                 offset vertically by a few hundredths so overlapping lines stay visible
    accuracy     held-out accuracy on the task being trained; fresh nets are new at every phase
Seeds share a colour across columns; solid = continued net, dashed = fresh nets. Grey lines mark task switches.

Usage (from plasticity/)
    uv run python -m visualization_scripts.toy_models.plot_orthogonal_summary \
        --run-dirs results/orthogonal_tasks/<run a> results/orthogonal_tasks/<run b> ... --out <file.png>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import unit_geometry as geo  # noqa: E402
from visualization_scripts.common import set_style  # noqa: E402
from visualization_scripts.toy_models.plot_orthogonal_tasks import (  # noqa: E402
    COND_TITLE, TASK_TINTS, load_seed, seeds_in, task_name)

SEED_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#D55E00"]


def live_count(status, act_frac_train) -> np.ndarray:
    return ((status == geo.ACTIVE) & (act_frac_train > 0)).sum(1)


def run_label(cfg: dict) -> str:
    bias = "trainable hidden bias" if cfg["hidden_bias"] == "trainable" else "no hidden bias"
    if cfg.get("bias_init") == "uniform":
        bias = "random-init hidden bias"
    return f"{bias}, {cfg['steps_per_task']:,} steps/task"


def shade(ax, schedule, S):
    for q, task in enumerate(schedule):
        ax.axvspan(q * S, (q + 1) * S, color=TASK_TINTS[task], alpha=0.08, lw=0)
        if q:
            ax.axvline(q * S, color="0.6", lw=0.5)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dirs", "--run_dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    set_style()
    present = set()
    for run_dir in args.run_dirs:
        present |= set(load_seed(run_dir, seeds_in(run_dir)[0])[2])
    conds = [c for c in ("shifted", "centered") if c in present]
    fig, axes = plt.subplots(len(args.run_dirs), 2 * len(conds), figsize=(8 * len(conds), 2.9 * len(args.run_dirs)),
                             squeeze=False)
    for r, run_dir in enumerate(args.run_dirs):
        seeds = seeds_in(run_dir)
        for k, seed in enumerate(seeds):
            cfg, _, arrays = load_seed(run_dir, seed)
            color, off = SEED_COLORS[k % len(SEED_COLORS)], 0.06 * (k - (len(seeds) - 1) / 2)
            for c, cond in enumerate(conds):
                if cond not in arrays:
                    continue
                arr = arrays[cond]
                schedule, S, H = arr["schedule"].tolist(), int(arr["steps_per_task"]), cfg["width"]
                ax_live, ax_acc = axes[r, 2 * c], axes[r, 2 * c + 1]
                if k == 0:
                    for ax in (ax_live, ax_acc):
                        shade(ax, schedule, S)
                        ax.set_xlim(0, len(schedule) * S)
                    ax_live.set_ylim(-0.3, H + 0.3)
                    ax_live.set_yticks(range(H + 1))
                    ax_acc.set_ylim(0.4, 1.02)
                    ax_live.set_ylabel(f"{run_label(cfg)}\nlive units", fontsize=8)
                    ax_acc.set_ylabel("held-out accuracy")
                    if r == 0:
                        ax_live.set_title(f"{COND_TITLE[cond]}: live units of {H}", fontsize=9)
                        ax_acc.set_title(f"{COND_TITLE[cond]}: accuracy on the current task", fontsize=9)
                    if r == len(args.run_dirs) - 1:
                        ax_live.set_xlabel("training step")
                        ax_acc.set_xlabel("training step")
                ax_live.step(arr["step"], live_count(arr["status"], arr["act_frac_train"]) + off, where="post",
                             color=color, lw=1.6, label=f"seed {seed}, continued")
                fp, fs = arr["fresh_phase"], arr["fresh_step"]
                fresh_live = live_count(arr["fresh_status"], arr["fresh_act_frac_train"]) + off
                fresh_acc = arr["fresh_test_acc"][np.arange(len(fs)), arr["fresh_task"]]
                for q in range(len(schedule)):
                    m = fp == q
                    ax_live.step(fs[m], fresh_live[m], where="post", color=color, lw=0.9, ls="--", alpha=0.7,
                                 label=f"seed {seed}, fresh" if q == 0 else None)
                    ax_acc.plot(fs[m], fresh_acc[m], color=color, lw=0.9, ls="--", alpha=0.7)
                ax_acc.plot(arr["step"], arr["test_acc"][np.arange(len(arr["step"])), arr["task"]], color=color,
                            lw=1.4)
    axes[0, 0].legend(fontsize=6.5, loc="lower left", ncol=2)
    names = " / ".join(task_name(t) for t in sorted(set(schedule)))
    shape = {"stripe": " (stripes)", "xor": " (XOR)"}.get(cfg.get("label_rule"), "")
    fig.suptitle(f"Tasks {names}{shape}, input dim {cfg['input_dim']}, width {cfg['width']}, {len(schedule) - 1} switches"
                 " — solid: continued net, dashed: fresh net (new each phase)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
