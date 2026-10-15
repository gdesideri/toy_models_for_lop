"""
visualization_scripts/cluster_factors.py — What the cluster factors of a dataset look like
(data/cluster_factors.py), in two figures.

Figure 1: one row per factor, its clusters side by side
    Each cluster shows the 3 train images closest (in pixel space) to the mean image of its
    members (the mean of up to 2000 of them). Under each cluster: its id and its share of the
    train set. The id is coloured by the label a feature-parity task gives it: even id ->
    label 1, odd id -> label 0. The row label gives the factor name and the fraction of
    train images with an even id (class balance).

Figure 2: normalised mutual information between the factors, and with the dataset's class
    (F + 1) x (F + 1) heatmap: 0 = independent, 1 = one determines the other.

A table (class balance, cluster sizes, NMI with the class, largest NMI with another factor)
is printed.

Run with:
    uv run python -m visualization_scripts.cluster_factors --dataset mnist --factors 6 --clusters 8
    uv run python -m visualization_scripts.cluster_factors --dataset cifar10

Writes results/cluster_factors_<dataset>_f<F>_c<C>.png and ..._nmi.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from data.cluster_factors import cluster_cache_path, nmi, with_cluster_factors
from data.download import get_dataset
from visualization_scripts.common import PALETTE, RESULTS, bold_suptitle, save, set_style

PER_CLUSTER = 3     # example images per cluster
MEAN_OF = 2000      # members averaged to get a cluster's mean image


# ── Images ────────────────────────────────────────────────────────────────────

def to_display(x):
    """A (C, H, W) tensor as an image matplotlib can show: (H, W) or (H, W, 3) in [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x[0] if x.shape[0] == 1 else x.transpose(1, 2, 0)


def strip(images, gap=2, fill=1.0):
    """Images side by side, separated by `gap` pixels of value `fill`."""
    h = images[0].shape[0]
    pad = np.full((h, gap) + images[0].shape[2:], fill)
    return np.concatenate([part for im in images for part in (im, pad)][:-1], axis=1)


def load_images(dataset):
    """All images of `dataset` as a (N, C, H, W) float32 array."""
    return np.stack([np.asarray(dataset[i][0], dtype=np.float32) for i in range(len(dataset))])


def cluster_examples(images, members, k=PER_CLUSTER, seed=0):
    """Indices of the k members closest to the mean image of (up to MEAN_OF of) the members."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(members, min(MEAN_OF, len(members)), replace=False)
    mean = images[sample].reshape(len(sample), -1).mean(0)
    d = ((images[members].reshape(len(members), -1) - mean) ** 2).sum(1)
    return members[np.argsort(d)[:k]]


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_examples(train, images, n_clusters, out, name):
    n_factors = train.latent_idx.shape[1]
    sizes = train.cluster_sizes / train.cluster_sizes.sum(1, keepdims=True)
    h, w = images.shape[2:]
    gap = max(2, w // 4)                # between the images of a group
    wide = 3 * gap                      # between the groups
    group_w = PER_CLUSTER * w + (PER_CLUSTER - 1) * gap
    channels = images.shape[1]

    fig_w = max(8.0, n_clusters * 1.15 + 1.6)
    fig, axes = plt.subplots(n_factors, 1, figsize=(fig_w, n_factors * 1.0 + 1.3), squeeze=False)
    for f, ax in enumerate(axes[:, 0]):
        parts = []
        for c in range(n_clusters):
            members = np.flatnonzero(train.latent_idx[:, f] == c)
            group = strip([to_display(images[i]) for i in cluster_examples(images, members)], gap=gap)
            parts += [group, np.ones((h, wide) + group.shape[2:])]
        ax.imshow(np.concatenate(parts[:-1], axis=1), cmap="gray", vmin=0, vmax=1)
        ax.set_yticks([])
        ax.grid(False)
        ax.tick_params(axis="x", which="both", length=0, pad=2)
        centers = [c * (group_w + wide) + group_w / 2 - 0.5 for c in range(n_clusters)]
        ax.set_xticks(centers)
        ax.set_xticklabels([f"{c}\n{100 * sizes[f, c]:.0f}%" for c in range(n_clusters)], fontsize=8)
        for c, label in enumerate(ax.get_xticklabels()):
            label.set_color(PALETTE[0] if c % 2 == 0 else PALETTE[1])
            label.set_fontweight("bold")
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_ylabel(f"{train.factor_names[f]}\neven {100 * train.class_balance[f]:.0f}%",
                      rotation=0, ha="right", va="center", fontsize=9, labelpad=8)
    bold_suptitle(fig, f"Cluster factors — {name}: {n_clusters} clusters per factor, "
                       f"{PER_CLUSTER} images nearest each cluster's mean")
    fig.legend([Patch(color=PALETTE[0]), Patch(color=PALETTE[1])],
               ["even id (label 1)", "odd id (label 0)"], loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=0.6)
    return save(fig, out)


def plot_nmi(matrix, names, out, title):
    n = len(names)
    set_style()
    fig, ax = plt.subplots(figsize=(0.8 * n + 2.5, 0.8 * n + 1.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > 0.6 else "black")
    ax.axhline(n - 1.5, color="black", lw=0.8)
    ax.axvline(n - 1.5, color="black", lw=0.8)
    fig.colorbar(im, ax=ax, label="NMI", shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return save(fig, out)


def class_labels(train):
    return np.asarray(train.dataset.targets)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--factors", type=int, default=6)
    p.add_argument("--clusters", type=int, default=8)
    p.add_argument("--dims", type=int, default=50)
    args = p.parse_args()
    F, C = args.factors, args.clusters

    set_style()
    train, _ = with_cluster_factors(*get_dataset(args.dataset), F, C, args.dims,
                                    cache_path=cluster_cache_path(args.dataset, F, C, args.dims))
    images = load_images(train)
    RESULTS.mkdir(exist_ok=True)
    stem = RESULTS / f"cluster_factors_{args.dataset}_f{F}_c{C}"
    plot_examples(train, images, C, stem.with_name(stem.name + ".png"), args.dataset)

    labels = class_labels(train)
    columns = [train.latent_idx[:, f] for f in range(F)] + [labels]
    full = np.array([[nmi(a, b) for b in columns] for a in columns])
    names = list(train.factor_names) + ["class"]
    plot_nmi(full, names, stem.with_name(stem.name + "_nmi.png"),
             f"NMI between the cluster factors — {args.dataset}")

    print(f"\n{'factor':<8}{'even %':>7}  {'cluster sizes %':<{4 * C}}{'NMI class':>10}{'max NMI other':>15}")
    for f in range(F):
        sizes = " ".join(f"{100 * s:3.0f}" for s in train.cluster_sizes[f] / train.cluster_sizes[f].sum())
        others = np.delete(full[f, :F], f)
        print(f"{names[f]:<8}{100 * train.class_balance[f]:>7.1f}  {sizes:<{4 * C}}"
              f"{full[f, F]:>10.3f}{(others.max() if len(others) else 0.0):>15.3f}")


if __name__ == "__main__":
    main()
