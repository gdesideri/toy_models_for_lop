"""
visualization_scripts/azimuth_width_sweep.py — Test accuracy vs hidden width on
Shapes3D object_azimuth (15 classes, 100% permuted pixels), one point per seed.

Run with:  uv run python -m visualization_scripts.azimuth_width_sweep
Reads    : shapes3d_perm100/shapes3d_mlp_2x<w>_perm100_object_azimuth_seed*/summary.json
           (in results/, else its vault copy)
Writes   : results/shapes3d_perm100/azimuth_width_sweep.png
"""
from __future__ import annotations

import glob
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization_scripts import common

RUNS = "shapes3d_perm100"
SUMMARIES = "shapes3d_mlp_2x*_perm100_object_azimuth_seed*/summary.json"
CHANCE = 1 / 15


def load():
    acc = {}
    for p in glob.glob(str(common.results_dir(RUNS, SUMMARIES) / SUMMARIES)):
        w = int(re.search(r"mlp_2x(\d+)_", p).group(1))
        acc.setdefault(w, []).append(json.load(open(p))["final_test_acc"])
    return dict(sorted(acc.items()))


def main():
    acc = load()
    widths = list(acc)
    common.set_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.5))

    rng = np.random.default_rng(0)
    for w in widths:
        x = w + rng.uniform(-0.8, 0.8, len(acc[w]))
        ax0.scatter(x, acc[w], color=common.PALETTE[0], marker=common.MARKERS[0],
                    alpha=0.6, s=40, zorder=3)
    ax0.plot(widths, [np.mean(acc[w]) for w in widths], **common.style_for(1),
             marker=common.MARKERS[1], label=r"mean over seeds")
    ax0.axhline(CHANCE, **common.style_for(2), label=r"chance $1/15$")
    ax0.set_xlabel(r"hidden units per layer")
    ax0.set_ylabel(r"final test accuracy")
    ax0.set_ylim(0, 1.05)
    ax0.set_xticks(widths)
    ax0.set_title(r"Per-seed outcome")
    ax0.legend(loc="lower right")

    n = np.array([len(acc[w]) for w in widths])
    dead = np.array([sum(a < 0.10 for a in acc[w]) for w in widths]) / n
    solved = np.array([sum(a > 0.95 for a in acc[w]) for w in widths]) / n
    ax1.plot(widths, solved, **common.style_for(0), marker=common.MARKERS[0],
             label=r"solved ($> 0.95$)")
    ax1.plot(widths, 1 - solved - dead, **common.style_for(1), marker=common.MARKERS[1],
             label=r"partial")
    ax1.plot(widths, dead, **common.style_for(2), marker=common.MARKERS[2],
             label=r"at chance ($< 0.10$)")
    ax1.set_xlabel(r"hidden units per layer")
    ax1.set_ylabel(r"fraction of seeds")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xticks(widths)
    ax1.set_title(rf"Outcome fractions ($n = {int(n.min())}$ seeds per width)")
    ax1.legend(loc="center right")

    common.bold_suptitle(fig, r"Shapes3D object azimuth, $100\%$ permuted, MLP $2 \times w$")
    fig.tight_layout()
    out = common.RESULTS / RUNS
    out.mkdir(parents=True, exist_ok=True)
    common.save(fig, out / "azimuth_width_sweep.png")


if __name__ == "__main__":
    main()
