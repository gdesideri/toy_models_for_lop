"""
plot_orthogonal_summary.py — one compact figure over several orthogonal_tasks runs: live hidden units and
held-out accuracy over training, continued net vs fresh nets, every seed.

One row per run, two columns:
    live units   units active on at least one train input; seeds are offset vertically by a few hundredths
                 so overlapping lines stay visible
    accuracy     held-out accuracy on the task being trained; fresh nets are new at every phase
Seeds share a colour across columns; solid = continued net, dashed = fresh nets. Grey lines mark switches.

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

from orthogonal_tasks.config import task_name  # noqa: E402
from orthogonal_tasks.summary import live_count, load_seed, seeds_in  # noqa: E402
from visualization_scripts.common import set_style  # noqa: E402
from visualization_scripts.toy_models.orthogonal_common import SEED_COLORS, shade_tasks  # noqa: E402

RULE_TITLE = {"halfspace": "", "stripe": " (stripes)", "xor": " (XOR)"}


def run_label(cfg) -> str:
    return f"{cfg.steps_per_task:,} steps/task" + ("" if cfg.bias_init == "uniform" else f", {cfg.bias_init} bias init")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    set_style()
    n = len(args.run_dirs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.9 * n), squeeze=False)
    for r, run_dir in enumerate(args.run_dirs):
        ax_live, ax_acc = axes[r]
        seeds = seeds_in(run_dir)
        for k, seed in enumerate(seeds):
            cfg, _, arr = load_seed(run_dir, seed)
            schedule, S, H = cfg.schedule, cfg.steps_per_task, cfg.width
            color, off = SEED_COLORS[k % len(SEED_COLORS)], 0.06 * (k - (len(seeds) - 1) / 2)
            if k == 0:
                for ax in (ax_live, ax_acc):
                    shade_tasks(ax, schedule, S)
                    ax.set_xlim(0, len(schedule) * S)
                ax_live.set_ylim(-0.3, H + 0.3)
                ax_live.set_yticks(range(H + 1))
                ax_acc.set_ylim(0.4, 1.02)
                ax_live.set_ylabel(f"{run_label(cfg)}\nlive units", fontsize=8)
                ax_acc.set_ylabel("held-out accuracy")
                if r == 0:
                    ax_live.set_title(f"live units of {H}", fontsize=9)
                    ax_acc.set_title("accuracy on the current task", fontsize=9)
                if r == n - 1:
                    ax_live.set_xlabel("training step")
                    ax_acc.set_xlabel("training step")
            ax_live.step(arr["step"], live_count(arr["act_frac_train"]) + off, where="post", color=color, lw=1.6,
                         label=f"seed {seed}, continued")
            ax_acc.plot(arr["step"], arr["test_acc"][np.arange(len(arr["step"])), arr["task"]], color=color, lw=1.4)
            fresh_live = live_count(arr["fresh_act_frac_train"]) + off
            fresh_acc = arr["fresh_test_acc"][np.arange(len(arr["fresh_step"])), arr["fresh_task"]]
            for q in range(len(schedule)):
                m = arr["fresh_phase"] == q
                ax_live.step(arr["fresh_step"][m], fresh_live[m], where="post", color=color, lw=0.9, ls="--",
                             alpha=0.7, label=f"seed {seed}, fresh" if q == 0 else None)
                ax_acc.plot(arr["fresh_step"][m], fresh_acc[m], color=color, lw=0.9, ls="--", alpha=0.7)
    axes[0, 0].legend(fontsize=6.5, loc="lower left", ncol=2)
    names = " / ".join(task_name(t) for t in sorted(set(schedule)))
    fig.suptitle(f"Tasks {names}{RULE_TITLE[cfg.label_rule]}, input dim {cfg.input_dim}, cloud {cfg.scales}, "
                 f"width {cfg.width}, {len(schedule) - 1} switches — solid: continued net, dashed: fresh net",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
