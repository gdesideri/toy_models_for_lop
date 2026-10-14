"""
visualization_scripts/factor_rule_pretrain_factors.py — How well each Shapes3D factor is
classified from the pretrained encoder of run_factor_rule_stream.py, one bar per factor
(mean over seeds, dots = seeds), with the chance level 1 / K of each factor.

    --measure head    test accuracy of the six temporary factor heads at the end of the
                      pretraining (stage 1), read from each run's checkpoints/encoder.pt
    --measure probe   test accuracy of the linear probes on the frozen representation
                      (stage 3), read from each run's representation_probe_results.csv

Run with:  uv run python -m visualization_scripts.factor_rule_pretrain_factors [results dir] [--measure head|probe]
           (default dir: shapes3d_factor_rule_stream in results/, else its vault copy)
Writes   : results/factor_rule_pretrain_factors.png   (head)
           results/factor_rule_probe_factors.png      (probe)
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.download import SHAPES3D_FACTOR_NAMES, SHAPES3D_FACTOR_N_VALUES
from visualization_scripts import common

RUNS = "shapes3d_factor_rule_stream"
SOURCES = {"head": "factorrule_*_seed*/checkpoints/encoder.pt",
           "probe": "factorrule_*_seed*/representation_probe_results.csv"}

TITLE_SIZE = 22
LABEL_SIZE = 22
TICK_SIZE = 18


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs_dir", nargs="?", type=Path, default=None)
    p.add_argument("--measure", choices=("head", "probe"), default="head")
    args = p.parse_args()

    factors = list(SHAPES3D_FACTOR_NAMES)
    runs_dir = args.runs_dir or common.results_dir(RUNS, SOURCES[args.measure])
    files = sorted(glob.glob(str(runs_dir / SOURCES[args.measure])))
    if args.measure == "head":
        acc = [[torch.load(f, map_location="cpu", weights_only=False)["meta"]["per_factor"]["test"][name]["acc"]
                for name in factors] for f in files]
        title, out_name = "Encoder pretraining: accuracy of each factor head", "factor_rule_pretrain_factors.png"
    else:
        acc = []
        for f in files:
            rows = {r["factor"]: float(r["test_accuracy"]) for r in csv.DictReader(open(f))}
            acc.append([rows[name] for name in factors])
        title, out_name = "Frozen encoder: linear-probe accuracy of each factor", "factor_rule_probe_factors.png"
    if not files:
        raise SystemExit(f"no {args.measure} results under {runs_dir}")
    acc = np.array(acc)                                             # (seeds, factors)
    chance = 1.0 / np.array(SHAPES3D_FACTOR_N_VALUES)
    for name, a, c in zip(factors, acc.T, chance):
        print(f"{name:15s} mean {a.mean():.3f}  seeds {np.round(a, 3).tolist()}  chance {c:.3f}")

    common.set_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(factors))
    ax.bar(x, acc.mean(0), width=0.62, color=common.CMAP(0.45), zorder=2)
    ax.plot(np.repeat(x, len(acc)), acc.T.ravel(), linestyle="none", marker="o", markersize=7,
            color="0.15", markeredgecolor="white", markeredgewidth=1.2, zorder=4, label="seeds")
    ax.hlines(chance, x - 0.31, x + 0.31, color="0.15", linestyle=":", linewidth=2.2, zorder=3, label="chance")
    for xi, a in zip(x, acc.mean(0)):
        ax.text(xi, 1.03, f"{a:.2f}", ha="center", va="bottom", fontsize=TICK_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name.replace('_', ' ').replace(' ', chr(10))}\n({k})"
                        for name, k in zip(factors, SHAPES3D_FACTOR_N_VALUES)])
    ax.set_xlabel("factor (number of classes)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("test accuracy", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.tick_params(axis="x", which="both", length=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, fontsize=TICK_SIZE, frameon=False)
    common.bold_suptitle(fig, title, fontsize=TITLE_SIZE)
    fig.tight_layout()
    common.save(fig, common.RESULTS / out_name)


if __name__ == "__main__":
    main()
