"""
visualization_scripts/feature_pair_distance.py — The 6x6 representation-distance
matrix of experiments/launch_scripts/submit_feature_pairs_repr_distance_array.sh.

Row = feature task 1 was trained on (the body), column = feature task 2's head is
retrained on. Cells average only over seeds where BOTH from-scratch models solved
their task (test acc >= 0.95): elsewhere the distance measures a failed model.

Run with:  uv run python -m visualization_scripts.feature_pair_distance [results dir]
           (default shapes3d_reprdist_features in results/, else its vault copy;
            e.g. results/shapes3d_reprdist_features_sgd for the SGD sweep)
Reads    : <results dir>/*/summary.json
Writes   : <results dir>/feature_pair_distance.{png,json}
           (default dir: results/shapes3d_reprdist_features/feature_pair_distance.{png,json})
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization_scripts import common

RUNS = "shapes3d_reprdist_features"
LABELS = [r"floor hue", r"wall hue", r"object hue", r"size", r"type", r"azimuth"]
SOLVED = 0.95


def load(root):
    runs = [json.load(open(p)) for p in glob.glob(str(root / "*" / "summary.json"))]
    keys = ["repr_distance", "task1_body_task2_head_test_acc", "zero_shot_task2_train_loss"]
    mats = {k: np.full((6, 6), np.nan) for k in keys}
    n_all, n_ok = np.zeros((6, 6), int), np.zeros((6, 6), int)
    cells = {}
    for r in runs:
        cells.setdefault((r["task1_target_factor"], r["task2_target_factor"]), []).append(r)
    for (a, b), rs in cells.items():
        ok = [r for r in rs if min(r["task1_model_test_acc"], r["task2_model_test_acc"]) >= SOLVED]
        n_all[a, b], n_ok[a, b] = len(rs), len(ok)
        for k in keys:
            if ok:
                mats[k][a, b] = np.mean([r[k] for r in ok])
    return runs, mats, n_all, n_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", type=Path, default=None)
    arg = p.parse_args().root
    root = arg.resolve() if arg else common.results_dir(RUNS, "*/summary.json")
    out_dir = arg.resolve() if arg else common.RESULTS / RUNS
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, mats, n_all, n_ok = load(root)
    common.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    panels = [("repr_distance", r"Representation distance", (0.35, 0.5)),
              ("task1_body_task2_head_test_acc", r"Retrained-head test accuracy on task 2", (0.5, 0.65))]
    for ax, (key, title, (lo, hi)) in zip(axes, panels):
        m = np.ma.masked_invalid(mats[key])
        im = ax.imshow(m, cmap=common.CMAP, vmin=lo, vmax=hi)
        for a in range(6):
            for b in range(6):
                if a == b:
                    ax.text(b, a, r"--", ha="center", va="center", fontsize=12)
                elif not np.isnan(mats[key][a, b]):
                    v = mats[key][a, b]
                    dark = (v - lo) / (hi - lo) < 0.5
                    ax.text(b, a, rf"${v:.3f}$" + "\n" + rf"$n={n_ok[a, b]}$", ha="center", va="center",
                            fontsize=10, color="white" if dark else "black")
        ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(6), LABELS)
        ax.set_xlabel(r"task 2 (head retrained on)")
        ax.set_ylabel(r"task 1 (body trained on)")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    opt = json.load(open(next(root.glob("*/model_task1/args.json"))))["optimizer"]
    common.bold_suptitle(fig, r"Shapes3D feature-parity pairs, MLP $2 \times 30$, " + opt.upper()
                            + r" (seeds where both models solved)")
    fig.tight_layout()
    print(f"{len(runs)} runs")
    common.save(fig, out_dir / "feature_pair_distance.png")
    json.dump({"factors": LABELS, "n_runs": n_all.tolist(), "n_solved": n_ok.tolist(),
               **{k: np.where(np.isnan(v), None, v).tolist() for k, v in mats.items()}},
              open(out_dir / "feature_pair_distance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
