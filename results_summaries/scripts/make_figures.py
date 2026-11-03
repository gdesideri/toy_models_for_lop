"""The summary figures that did not exist before results_summaries/, with their data.

    ~/notebooks/.venv/bin/python results_summaries/scripts/make_figures.py

(needs pandas, which `plasticity/` does not depend on; run `summarize_streams.py`
first, three figures read its CSVs.) Writes into `results_summaries/`:

    figures/02_width_sweep_collapse.png             collapse task vs hidden width, Shapes3D feature stream
    figures/04_streams_vs_fresh.png                 600-task streams against single-task fresh baselines
    figures/08_feature_acquisition_gaps.png         Result 1: full adaptation minus best frozen decoder
    figures/09_history_acquisition_deficits.png     Result 2: acquisition deficit by history
    csv/width_sweep_collapse.csv, csv/streams_vs_fresh.csv,
    csv/feature_acquisition_gaps.csv, csv/history_acquisition_deficits.csv

Style is the project's own (`visualization_scripts.common.set_style`: Nature
preset, Okabe-Ito). Series colours are taken in a fixed order validated with
`dataviz/scripts/validate_palette.js` (all checks pass; contrast WARN for the
light hues), and every series also carries a marker and a direct label or
legend entry.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plasticity"))
from visualization_scripts.common import set_style  # noqa: E402

CSV = ROOT / "results_summaries" / "csv"
FIG = ROOT / "results_summaries" / "figures"
VAULT = Path(os.environ.get("HPCVAULT", "/home/vault/v170be/v170be10")) / "plasticity_results"
HNVME = Path("/hnvme/workspace/v170be10-plasticity_results")

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED = "#222222", "#6b6b6b"
COLLAPSE_ACC, COLLAPSE_WINDOW = 0.75, 10


def _save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / name}")


def _collapse_task(run: Path) -> int | None:
    """First task whose 10-task moving average of test accuracy is below 0.75."""
    accs = []
    for p in run.glob("task_*/summary.json"):
        s = json.loads(p.read_text())
        accs.append((s["task"], s["final_test_acc"]))
    acc = np.array([a for _, a in sorted(accs)])
    ma = np.convolve(acc, np.ones(COLLAPSE_WINDOW) / COLLAPSE_WINDOW, mode="valid")
    below = np.nonzero(ma < COLLAPSE_ACC)[0]
    return int(below[0]) if below.size else None


def width_sweep() -> None:
    runs = sorted(glob.glob(str(HNVME / "shapes3d_stream_features_width_sgd" / "stream_*")))
    runs += sorted(glob.glob(str(VAULT / "shapes3d_stream_features_sgd" / "stream_*")))  # the width-30 original
    rows = [{"width": int(re.search(r"mlp_2x(\d+)", r).group(1)), "seed": int(r[-1]),
             "collapse_task": _collapse_task(Path(r))} for r in runs]
    d = pd.DataFrame(rows).sort_values(["width", "seed"])
    d.to_csv(CSV / "width_sweep_collapse.csv", index=False)
    m = d.groupby("width")["collapse_task"].mean()
    slope = np.polyfit(m.index, m.values, 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(d["width"], d["collapse_task"], s=45, color=COLORS[0], alpha=0.55, edgecolor="white",
               linewidth=1, zorder=3, label="one seed")
    ax.plot(m.index, m.values, color=COLORS[0], lw=2, marker="o", ms=8, mec="white", mew=1.5,
            zorder=4, label="mean of 3 seeds")
    xs = np.array([0, 230])
    ax.plot(xs, np.polyval(slope, xs), color=MUTED, lw=1, zorder=2,
            label=f"fit: {slope[0]:.2f} tasks per unit {slope[1]:+.0f}")
    ax.set_xlim(0, 230)
    ax.set_ylim(0, 270)
    ax.set_xlabel("hidden width (2 hidden layers)")
    ax.set_ylabel("collapse task\n(10-task mean test acc < 0.75)")
    ax.set_title("Shapes3D parity feature-switch stream: collapse comes later in wider nets, always", fontsize=12)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, "02_width_sweep_collapse.png")


def streams_vs_fresh() -> None:
    g = pd.read_csv(CSV / "stream_groups.csv")
    pairs = [  # (label, stream group, single-task group)
        ("CIFAR-10 class attributes, 2x120", "cifar10_class_stream_mlp_2x120", "cifar10_class_single_mlp_2x120"),
        ("CIFAR-10 cluster factors, 2x120", "cifar10_cluster_stream_mlp_2x120", "cifar10_cluster_single_mlp_2x120"),
        ("MNIST cluster factors, 2x30 *", "mnist_cluster_stream_mlp_2x30", "mnist_cluster_single_mlp_2x30"),
        ("MNIST cluster factors, 2x120 *", "mnist_cluster_stream_mlp_2x120", "mnist_cluster_single_mlp_2x120"),
        ("PCA-cosine CIFAR-10 parity, 2x120", "pca_cos_cifar10_parity_mlp_2x120", "pca_cos_cifar10_parity_single_mlp_2x120"),
        ("PCA-cosine CIFAR-10 upper half, 2x120", "pca_cos_cifar10_upper_half_mlp_2x120", "pca_cos_cifar10_upper_half_single_mlp_2x120"),
        ("PCA-cosine MNIST parity, 2x30", "pca_cos_mnist_parity_mlp_2x30", "pca_cos_mnist_parity_single_mlp_2x30"),
        ("PCA-cosine MNIST upper half, 2x30", "pca_cos_mnist_upper_half_mlp_2x30", "pca_cos_mnist_upper_half_single_mlp_2x30"),
    ]
    rows = []
    for label, s, f in pairs:
        gs = g[g["group"] == s].iloc[0]
        gf = g[g["group"] == f].iloc[0]
        rows.append({"benchmark": label, "stream_group": s, "single_group": f,
                     "tasks_done": int(gs["tasks_done_max"]), "fresh_test_acc": gf["test_acc_first5"],
                     "stream_first5": gs["test_acc_first5"], "stream_last5": gs["test_acc_last5"],
                     "stream_train_last5": gs["train_acc_last5"]})
    d = pd.DataFrame(rows)
    d.to_csv(CSV / "streams_vs_fresh.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(d))[::-1]
    series = [("fresh single-task network", "fresh_test_acc", COLORS[0], "o"),
              ("stream, first 5 visits per factor", "stream_first5", COLORS[1], "s"),
              ("stream, last 5 visits per factor", "stream_last5", COLORS[3], "D")]
    # the three series often coincide; a small vertical offset keeps each visible
    offsets = [0.2, 0.0, -0.2]
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r["stream_first5"], r["stream_last5"]], [yi, yi - 0.2], color=MUTED, lw=1, zorder=1)
    for (name, col, color, marker), off in zip(series, offsets):
        ax.scatter(d[col], y + off, color=color, marker=marker, s=55, edgecolor="white", linewidth=1.2,
                   zorder=3, label=name)
    ax.set_yticks(y, d["benchmark"])
    ax.set_xlabel("test accuracy (mean over factors and seeds)")
    ax.set_xlim(0.45, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=3, frameon=False, fontsize=9)
    ax.set_title("600-task streams vs fresh networks: only MNIST upper-half collapses", fontsize=12)
    ax.annotate("* MNIST cluster streams stopped at ~285 of 600 tasks", (0, 0), xycoords="axes fraction",
                xytext=(0, -62), textcoords="offset points", fontsize=9, color=MUTED)
    fig.tight_layout()
    _save(fig, "04_streams_vs_fresh.png")


def feature_acquisition() -> None:
    runs = [
        ("synthetic, mlp_2x32", "synthetic_mlp_2x32_synthetic_width32"),
        ("synthetic, mlp_2x64 (discovery)", "synthetic_mlp_2x64_synthetic_discovery"),
        ("synthetic, mlp_2x64 (confirmation)", "synthetic_mlp_2x64_synthetic_confirmation"),
        ("synthetic, mlp_2x256", "synthetic_mlp_2x256_synthetic_width256"),
        ("Shapes3D, mlp_3x24", "shapes3d_mlp_3x24_shapes3d_width24"),
        ("Shapes3D, mlp_3x32", "shapes3d_mlp_3x32_shapes3d_width32"),
        ("Shapes3D, mlp_3x256", "shapes3d_mlp_3x256_shapes3d_mlp"),
        ("Shapes3D, mlp_3x256, 32k-update decoders", "shapes3d_mlp_3x256_shapes3d_mlp_longdec"),
        ("Shapes3D, CNN", "shapes3d_cnn_shapes3d_cnn"),
    ]
    rows = []
    for label, tag in runs:
        f = glob.glob(str(VAULT / "feature_acquisition" / f"{tag}-*" / "comparisons.csv"))
        if not f:
            continue
        c = pd.read_csv(f[0])
        c = c[(c["comparison"] == "full_minus_frozen_extended") & (c["condition"] == "partial")
              & (c["task_kind"] == "factor") & (c["metric"] == "final_test_acc")]
        for _, r in c.iterrows():
            rows.append({"run": label, "tag": tag, "destination": r["task"], "gap_points": 100 * r["mean"],
                         "ci95_low": 100 * r["ci95_low"], "ci95_high": 100 * r["ci95_high"],
                         "n_seeds": r["n_seeds"]})
    d = pd.DataFrame(rows)
    d.to_csv(CSV / "feature_acquisition_gaps.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [l for l, _ in runs if l in set(d["run"])]
    y = np.arange(len(labels))[::-1]
    for yi, label in zip(y, labels):
        r = d[d["run"] == label]
        ax.hlines(yi, r["ci95_low"].min(), r["ci95_high"].max(), color=MUTED, lw=1, zorder=1)
        ax.scatter(r["gap_points"], [yi] * len(r), color=COLORS[0] if label.startswith("synthetic") else COLORS[2],
                   marker="o" if label.startswith("synthetic") else "^", s=50, edgecolor="white", linewidth=1.2,
                   zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("full adaptation minus best frozen decoder, final test accuracy (points)\n"
                  "one dot per new-factor destination; line = range of their 95% seed intervals")
    ax.set_title("Result 1: features a partially trained encoder lacks are only acquired by adapting it",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "08_feature_acquisition_gaps.png")


def history_acquisition() -> None:
    runs = [("synthetic, mlp_2x32", "synthetic_mlp_2x32_width32"),
            ("synthetic, mlp_2x64", "synthetic_mlp_2x64_discovery"),
            ("synthetic, mlp_2x256", "synthetic_mlp_2x256_width256"),
            ("Shapes3D, mlp_3x256", "shapes3d_mlp_3x256_discovery"),
            ("Shapes3D, mlp_2x512", "shapes3d_mlp_2x512_mlp2x512")]
    rows = []
    for label, tag in runs:
        f = glob.glob(str(VAULT / "history_acquisition" / f"{tag}-*" / "analysis" / "main-*" / "acquisition_stats.csv"))[0]
        s = pd.read_csv(f, usecols=["family", "condition", "age", "branch", "budget_name", "target", "metric", "mean"])
        s = s[(s["metric"] == "deficit_mean_acc") & (s["branch"] == "full") & (s["budget_name"].isin(["U", "UE"]))]
        s = s[s["age"] == s.groupby("condition")["age"].transform("max")]
        t = (s.groupby(["family", "condition", "age", "budget_name"])["mean"].mean().mul(100)
             .unstack("budget_name").reset_index().rename(columns={"U": "deficit_U", "UE": "deficit_UE"}))
        t.insert(0, "run", label)
        rows.append(t)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(CSV / "history_acquisition_deficits.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    conds = ["A_persistent_rho0", "A_persistent_rho0.25", "A_renewed_rho0.25", "A_persistent_rho0.75", "A_renewed_rho0.75"]
    names = ["clean repeat (rho 0)", "persistent, rho 0.25", "renewed, rho 0.25", "persistent, rho 0.75", "renewed, rho 0.75"]
    y = np.arange(len(conds))[::-1]
    for i, (label, _) in enumerate(runs):
        r = d[d["run"] == label].set_index("condition").reindex(conds)
        ax.scatter(r["deficit_U"], y + (i - 2) * 0.12, color=COLORS[i], marker=MARKERS[i], s=45,
                   edgecolor="white", linewidth=1, zorder=3, label=label)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, names)
    ax.set_xlabel("acquisition deficit vs fresh network\n(test accuracy points, first 2000 updates)")
    ax.set_title("(a) label-overwrite histories (family A)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = axes[1]
    for i, (label, _) in enumerate(runs):
        r = d[(d["run"] == label) & (d["family"] == "C")].copy()
        r["L"] = r["condition"].str.extract(r"C_L(\d+)").astype(int)
        r = r.sort_values("L")
        ax.plot(r["L"], r["deficit_U"], color=COLORS[i], marker=MARKERS[i], ms=7, lw=2, mec="white", mew=1.2)
    ax.set_xscale("log")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("block length L (updates per rule before switching)")
    ax.set_ylabel("deficit (points)")
    ax.set_title("(b) switching frequency (family C)")
    fig.suptitle("Result 2: which training histories slow later feature acquisition (discovery runs, 6 seeds)",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, "09_history_acquisition_deficits.png")


def main() -> None:
    set_style()
    FIG.mkdir(parents=True, exist_ok=True)
    width_sweep()
    streams_vs_fresh()
    feature_acquisition()
    history_acquisition()


if __name__ == "__main__":
    main()
