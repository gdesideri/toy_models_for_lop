"""
visualization_scripts/pca_components_described.py — The first six principal components of the raw
pixels of MNIST and CIFAR-10 (the factors of the PCA benchmark, data/pca_factors.py), one figure per
dataset, each component with a short description:

    image        the component: red = positive, blue = negative pixel weight (MNIST); for CIFAR-10
                 each channel's weights min-max scaled to RGB (light = positive, dark = negative)
    title        the component's share of the pixel variance
    text         below each component, what it encodes (the dataset summary is printed, not drawn)
    --shifts     adds a row above and below: the train mean image moved by -2 sigma and +2 sigma
                 along the component (sigma = the std of the train projections), in pixel space

PCA is fitted as the benchmark fits it (normalised, flattened train images; fit_pca).

Run with:
    uv run python -m visualization_scripts.pca_components_described                 # both datasets
    uv run python -m visualization_scripts.pca_components_described --dataset mnist
    uv run python -m visualization_scripts.pca_components_described --shifts        # with the +-2 sigma rows

Writes results/pca_components_<dataset>.png (with --shifts: pca_components_<dataset>_shifts.png)
"""
from __future__ import annotations

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.download import get_dataset
from data.pca_factors import fit_pca, project
from visualization_scripts.common import RESULTS, bold_suptitle, save, set_style

N_COMPONENTS = 6
SHIFT = 2.0                     # the mean image is moved by +-SHIFT sigma along a component

# the per-channel normalisation of data/download.py, undone for display
NORMALISATION = {"mnist": ((0.1307,), (0.3081,)),
                 "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}

TITLES = {"mnist": "MNIST: the first six principal components of the pixels",
          "cifar10": "CIFAR-10: the first six principal components of the pixels"}

SUMMARIES = {
    "mnist": "Flat spectrum (9.7 % down to 4.3 %): the six tasks are of similar difficulty. The components are "
             "stroke patterns tied to digit identity, so some projections are multimodal (one peak per digit group).",
    "cifar10": "Steep spectrum: pc0 carries 10x the variance of pc5, so pc4/pc5 tasks learn much more slowly. The "
               "components are smooth, low-frequency lighting and colour patterns, largely independent of the class.",
}

# read off the -2 sigma / +2 sigma images (and, for MNIST, the digits in the lowest / highest bins)
DESCRIPTIONS = {
    "mnist": [
        "A thin vertical '1' vs a wide round '0': how much ink, and how round. Bimodal: the 1s form "
        "a peak of their own.",
        "A top loop on a slanted stem ('9', '7') vs horizontal strokes stacked up ('2', '3').",
        "A long loop slanted along the diagonal vs a loop on top with a tail curving down ('9'-like).",
        "A '4'/'9'-like shape open on the left vs an S-shaped curve ('3', '5').",
        "A ring with an empty centre ('0') vs ink across the middle ('8', '4').",
        "Opposite diagonals: a tilted closed loop vs a stroke leaning the other way.",
    ],
    "cifar10": [
        "Overall brightness, weighted towards the borders: a dark scene vs a bright, sky-like "
        "background. Not a class feature.",
        "Top vs bottom: bright top over a dark bottom vs a dark top over light ground.",
        "Centre vs border: a dark object on a light background vs a light object on a dark one.",
        "Left vs right: the light side of the image.",
        "Colour, not brightness: a blue, cool cast (sky, water) vs an orange-brown, warm one "
        "(earth, fur).",
        "The first finer spatial pattern: an hourglass, the central vertical band vs the left and "
        "right sides.",
    ],
}


def to_pixels(x, dataset):
    """A normalised (C, H, W) image back in [0, 1] pixel space, as (H, W) or (H, W, 3)."""
    mean, std = (np.asarray(v)[:, None, None] for v in NORMALISATION[dataset])
    x = np.clip(x * std + mean, 0.0, 1.0)
    return x[0] if x.shape[0] == 1 else x.transpose(1, 2, 0)


def component_image(c):
    """(C, H, W) weights for display: (H, W) signed, or (H, W, 3) min-max scaled per channel."""
    if c.shape[0] == 1:
        return c[0]
    lo, hi = c.min(axis=(1, 2), keepdims=True), c.max(axis=(1, 2), keepdims=True)
    return ((c - lo) / (hi - lo + 1e-12)).transpose(1, 2, 0)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot(dataset, shifts=False):
    train, _ = get_dataset(dataset)
    shape = tuple(train[0][0].shape)
    mean, components, ratio = fit_pca(train, N_COMPONENTS)
    rows = ["minus", "component", "plus"] if shifts else ["component"]
    sigma = project(train, mean, components).std(axis=0) if shifts else None

    fig, axes = plt.subplots(len(rows), N_COMPONENTS, squeeze=False,
                             figsize=(2.7 * N_COMPONENTS, 7.6 if shifts else 2.9),
                             gridspec_kw=dict(hspace=0.08, wspace=0.18, top=0.93 if shifts else 0.84))
    for j in range(N_COMPONENTS):
        c = components[j].reshape(shape)
        for row, kind in enumerate(rows):
            ax = axes[row, j]
            if kind == "component" and shape[0] == 1:
                lim = np.abs(c).max()
                ax.imshow(component_image(c), cmap="RdBu_r", vmin=-lim, vmax=lim)
            elif kind == "component":
                ax.imshow(component_image(c))
            else:
                sign = -1 if kind == "minus" else 1
                shifted = (mean + sign * SHIFT * sigma[j] * components[j]).reshape(shape)
                ax.imshow(to_pixels(shifted, dataset), cmap="gray" if shape[0] == 1 else None, vmin=0, vmax=1)
            clean(ax)
        axes[0, j].set_title(f"pc{j}  ({100 * ratio[j]:.1f} % var)", fontweight="bold")
        axes[-1, j].text(0.5, -0.06, textwrap.fill(DESCRIPTIONS[dataset][j], 22), transform=axes[-1, j].transAxes,
                         ha="center", va="top", linespacing=1.3)
    if shifts:
        for row, label in enumerate([f"mean $-$ {SHIFT:g}$\\sigma$", "component", f"mean + {SHIFT:g}$\\sigma$"]):
            axes[row, 0].set_ylabel(label)
    bold_suptitle(fig, TITLES[dataset], y=1.0)
    print(f"\n{TITLES[dataset]}\n{SUMMARIES[dataset]}")
    suffix = "_shifts" if shifts else ""
    return save(fig, RESULTS / f"pca_components_{dataset}{suffix}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=list(NORMALISATION), nargs="+", default=list(NORMALISATION))
    parser.add_argument("--shifts", action="store_true", help="add the mean +- 2 sigma images above and below")
    args = parser.parse_args()
    set_style()
    for dataset in args.dataset:
        plot(dataset, args.shifts)


if __name__ == "__main__":
    main()
