"""
visualization_scripts/pca_factors.py — What the PCA factors of a dataset look like
(data/pca_factors.py), in three views.

--view bins (default): one row per principal component
    left   the component as an image (red = positive weight, blue = negative;
           for colour datasets the component is min-max scaled to RGB)
    middle histogram of the train images' projections on it, the K - 1 bin edges
           as vertical lines, coloured by the parity label (even bin -> 1, odd -> 0)
    right  one train image per bin, the one closest to the bin's median projection,
           from bin 0 (lowest projection) to bin K - 1

--view modes: one row per principal component, its modes
    middle histogram coloured by a two-component Gaussian mixture fitted to the
           projections, the smoothed density with its modes (triangles), the valley
           between the two highest modes (dashed) and the mixture's boundary (solid)
    right  the 4 images closest to each mixture component's mean
    A per-component table (modes, valley depth, splits) is printed.

--view joint: modes in the plane of a pair of components
    Every pair among the first --search_components components gets a two-component
    full-covariance Gaussian mixture (on a 10k-image subsample). Separation = how far
    apart its two clusters are (the mean term of the Bhattacharyya distance, so a narrow
    core inside a wide halo does not count as two modes); a pair scores by the separation
    it gains over the better of its two components alone, so a pair is not picked just
    for containing one bimodal component. Pairs whose smaller cluster holds < 10% of the
    images are skipped. The --pairs best pairs sharing no
    component are shown, refitted on all images: the two components, the 2-D histogram
    with the clusters' 2-sigma ellipses and boundary, and 4 images per cluster.
    The 10 best-scoring pairs are printed.

Run with:
    uv run python -m visualization_scripts.pca_factors --dataset mnist --components 6 --bins 8
    uv run python -m visualization_scripts.pca_factors --dataset mnist --view modes
    uv run python -m visualization_scripts.pca_factors --dataset mnist --view joint --pairs 3

Writes results/pca_factors_<dataset>_pc<n>_bins<K>.png, results/pca_modes_<dataset>_pc<n>.png
       or results/pca_joint_modes_<dataset>.png
"""
from __future__ import annotations

import argparse
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse, Patch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from data.download import get_dataset
from data.pca_factors import assign_bins, bin_edges, fit_pca, project
from visualization_scripts.common import PALETTE, RESULTS, bold_suptitle, save, set_style


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


def show_components(ax, comps, shape):
    """One or more components side by side: diverging red/blue for grey images, RGB otherwise."""
    comps = [c.reshape(shape) for c in comps]
    if shape[0] == 1:
        lim = max(np.abs(c).max() for c in comps)
        ax.imshow(strip([c[0] for c in comps], fill=0.0), cmap="RdBu_r", vmin=-lim, vmax=lim)
    else:
        ax.imshow(strip([to_display(c) for c in comps]))
    clean(ax)


def show_examples(ax, dataset, groups, shape):
    """Groups of example images side by side, a wider gap between groups."""
    gap = np.ones((shape[1], 6) + ((3,) if shape[0] == 3 else ()))
    parts = []
    for g, group in enumerate(groups):
        if g:
            parts.append(gap)
        parts.append(strip([to_display(dataset[i][0]) for i in group]))
    ax.imshow(np.concatenate(parts, axis=1), cmap="gray" if shape[0] == 1 else None)
    clean(ax)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def nearest(points, center, k=4):
    """Indices of the k rows of `points` closest to `center`."""
    d = ((np.atleast_2d(points.T).T - center) ** 2).reshape(len(points), -1).sum(1)
    return np.argsort(d)[:k]


# ── Modes ─────────────────────────────────────────────────────────────────────

def density_modes(p):
    """
    Modes of the 1-D distribution `p`: peaks of a smoothed 200-bin histogram, and
    the valley between the two highest (its depth = valley / smaller peak, and the
    fraction of `p` above it). Valley fields are None for a single mode.
    """
    counts, edges = np.histogram(p, bins=200)
    density = gaussian_filter1d(counts.astype(float), 3)
    grid = (edges[:-1] + edges[1:]) / 2
    peaks, _ = find_peaks(density, prominence=0.02 * density.max())
    out = dict(grid=grid, density=density, width=edges[1] - edges[0], modes=grid[peaks],
               valley=None, depth=None, above=None)
    if len(peaks) >= 2:
        a, b = sorted(peaks[np.argsort(density[peaks])[-2:]])
        v = a + np.argmin(density[a:b])
        out.update(valley=grid[v], depth=density[v] / min(density[a], density[b]),
                   above=float(np.mean(p > grid[v])))
    return out


def gmm2(p, iters=200):
    """Two-component 1-D Gaussian mixture by EM: (weights, means, stds), means ascending."""
    mu = np.quantile(p, [0.25, 0.75])
    sd = np.full(2, p.std() / 2)
    w = np.full(2, 0.5)
    for _ in range(iters):
        r = gmm_posterior(p, w, mu, sd)
        n = r.sum(0)
        w, mu = n / len(p), (r * p[:, None]).sum(0) / n
        sd = np.sqrt((r * (p[:, None] - mu) ** 2).sum(0) / n) + 1e-6
    order = np.argsort(mu)
    return w[order], mu[order], sd[order]


def gmm_posterior(p, w, mu, sd):
    logp = np.log(w) - np.log(sd) - 0.5 * ((p[:, None] - mu) / sd) ** 2
    r = np.exp(logp - logp.max(1, keepdims=True))
    return r / r.sum(1, keepdims=True)


def gmm2_2d(z, iters=200, seed=0):
    """Two-component full-covariance 2-D Gaussian mixture by EM (k-means++ init): (w, mu, cov)."""
    rng = np.random.default_rng(seed)
    first = z[rng.integers(len(z))]
    d = ((z - first) ** 2).sum(1)
    mu = np.stack([first, z[rng.choice(len(z), p=d / d.sum())]])
    cov = np.stack([np.cov(z.T)] * 2)
    w = np.full(2, 0.5)
    for _ in range(iters):
        r = gmm_posterior_2d(z, w, mu, cov)
        n = r.sum(0)
        w, mu = n / len(z), (r.T @ z) / n[:, None]
        for k in range(2):
            dz = z - mu[k]
            cov[k] = (r[:, k, None] * dz).T @ dz / n[k] + 1e-6 * np.eye(2)
    order = np.argsort(mu[:, 0])
    return w[order], mu[order], cov[order]


def gmm_posterior_2d(z, w, mu, cov):
    logp = np.empty((len(z), 2))
    for k in range(2):
        dz = z - mu[k]
        logp[:, k] = (np.log(w[k]) - 0.5 * np.linalg.slogdet(cov[k])[1]
                      - 0.5 * np.einsum("ni,ij,nj->n", dz, np.linalg.inv(cov[k]), dz))
    r = np.exp(logp - logp.max(1, keepdims=True))
    return r / r.sum(1, keepdims=True)


def separation(mu, cov):
    """Mean term of the Bhattacharyya distance between two Gaussians."""
    dm = mu[1] - mu[0]
    return float(dm @ np.linalg.inv((cov[0] + cov[1]) / 2) @ dm / 8)


def choose_pairs(proj, n_pairs, min_weight=0.1, n_sample=10_000, seed=0):
    """
    Pairs ranked by the separation their two clusters gain over the better of the two
    components alone (a pair containing one bimodal component would otherwise win just
    by that component). Returns (ranked list of (gain, separation, (i, j), minority
    weight), chosen pairs sharing no component).
    """
    sample = proj[np.random.default_rng(seed).choice(len(proj), min(n_sample, len(proj)), replace=False)]
    alone = []
    for i in range(proj.shape[1]):
        w, mu, sd = gmm2(sample[:, i], iters=100)
        alone.append(separation(mu[:, None], (sd ** 2)[:, None, None]))
    ranked = []
    for i, j in combinations(range(proj.shape[1]), 2):
        w, mu, cov = gmm2_2d(sample[:, [i, j]], iters=100, seed=seed)
        if w.min() >= min_weight:
            sep = separation(mu, cov)
            ranked.append((sep - max(alone[i], alone[j]), sep, (i, j), float(w.min())))
    ranked.sort(reverse=True)
    chosen, used = [], set()
    for _, _, pair, _ in ranked:
        if not used & set(pair):
            chosen.append(pair)
            used |= set(pair)
        if len(chosen) == n_pairs:
            break
    return ranked, chosen


# ── Views ─────────────────────────────────────────────────────────────────────

def plot_bins(args, train, shape, components, ratio, proj):
    n = args.components
    edges = bin_edges(proj[:, :n], args.bins)
    idx = assign_bins(proj[:, :n], edges)
    fig, axes = plt.subplots(n, 3, figsize=(16, 2.4 * n),
                             gridspec_kw={"width_ratios": [1, 3, 0.9 * args.bins]})
    for j in range(n):
        ax_comp, ax_hist, ax_ex = axes[j]
        show_components(ax_comp, [components[j]], shape)
        ax_comp.set_ylabel(f"pc{j}\n{100 * ratio[j]:.1f}% var")

        counts, fine = np.histogram(proj[:, j], bins=120)
        centers = (fine[:-1] + fine[1:]) / 2
        even = np.searchsorted(edges[j], centers, side="right") % 2 == 0
        ax_hist.bar(centers, counts, width=np.diff(fine), linewidth=0,
                    color=np.where(even, PALETTE[0], PALETTE[1]))
        for e in edges[j]:
            ax_hist.axvline(e, color="0.4", linewidth=0.8)
        ax_hist.set_yticks([])

        examples = []
        for b in range(args.bins):
            members = np.flatnonzero(idx[:, j] == b)
            examples.append(members[nearest(proj[members, j], np.median(proj[members, j]), 1)[0]])
        ax_ex.imshow(strip([to_display(train[i][0]) for i in examples]),
                     cmap="gray" if shape[0] == 1 else None)
        clean(ax_ex)
        ax_ex.set_xticks([b * (shape[2] + 2) + shape[2] / 2 for b in range(args.bins)],
                         [str(b) for b in range(args.bins)])
    axes[-1, 1].set_xlabel("projection on the component")
    axes[-1, 2].set_xlabel("bin")
    axes[0, 0].set_title("component")
    axes[0, 1].set_title("train projections and bin edges")
    axes[0, 2].set_title("one image per bin")
    fig.legend(handles=[Patch(color=PALETTE[0], label="even bin (label 1)"),
                        Patch(color=PALETTE[1], label="odd bin (label 0)")],
               loc="lower center", ncol=2, frameon=False)
    bold_suptitle(fig, f"PCA factors of {args.dataset}: {n} components, {args.bins} bins")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig, RESULTS / f"pca_factors_{args.dataset}_pc{n}_bins{args.bins}.png"


def plot_modes(args, train, shape, components, ratio, proj):
    n = args.components
    fig, axes = plt.subplots(n, 3, figsize=(16, 2.4 * n), gridspec_kw={"width_ratios": [1, 3, 7]})
    print(f"{'pc':>4} {'var%':>6} {'modes':>6} {'valley depth':>13} {'valley split':>13} {'GMM split':>10}")
    for j in range(n):
        ax_comp, ax_hist, ax_ex = axes[j]
        p = proj[:, j]
        show_components(ax_comp, [components[j]], shape)
        ax_comp.set_ylabel(f"pc{j}\n{100 * ratio[j]:.1f}% var")

        dm = density_modes(p)
        w, mu, sd = gmm2(p)
        high = gmm_posterior(dm["grid"], w, mu, sd).argmax(1) == 1
        counts, _ = np.histogram(p, bins=200)
        ax_hist.bar(dm["grid"], counts, width=dm["width"], linewidth=0,
                    color=np.where(high, PALETTE[1], PALETTE[0]))
        ax_hist.plot(dm["grid"], dm["density"], color="0.15", linewidth=1.2)
        ax_hist.plot(dm["modes"], np.interp(dm["modes"], dm["grid"], dm["density"]), "v",
                     color="0.15", markersize=8)
        if dm["valley"] is not None:
            ax_hist.axvline(dm["valley"], color="0.15", linestyle="--", linewidth=1)
        flips = np.flatnonzero(np.diff(high.astype(int)))
        for f in flips:
            ax_hist.axvline((dm["grid"][f] + dm["grid"][f + 1]) / 2, color="0.15", linewidth=1.2)
        split = float(np.mean(gmm_posterior(p, w, mu, sd).argmax(1) == 1))
        valley = "-" if dm["valley"] is None else f"{dm['depth']:.2f}"
        note = f"{len(dm['modes'])} mode{'s' * (len(dm['modes']) > 1)}, GMM split {100 * (1 - split):.0f}/{100 * split:.0f}"
        if dm["valley"] is not None:
            note += f"\nvalley depth {dm['depth']:.2f}"
        ax_hist.text(0.98, 0.95, note, transform=ax_hist.transAxes, ha="right", va="top", fontsize=9,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
        ax_hist.set_yticks([])
        above = dm["above"]
        valley_split = "-" if above is None else f"{100 * (1 - above):.0f}/{100 * above:.0f}"
        gmm_split = f"{100 * (1 - split):.0f}/{100 * split:.0f}"
        print(f"{'pc' + str(j):>4} {100 * ratio[j]:6.1f} {len(dm['modes']):6d} {valley:>13} "
              f"{valley_split:>13} {gmm_split:>10}")

        show_examples(ax_ex, train, [nearest(p, mu[0]), nearest(p, mu[1])], shape)
    axes[-1, 1].set_xlabel("projection on the component")
    axes[0, 0].set_title("component")
    axes[0, 1].set_title("projections, modes and 2-component mixture")
    axes[0, 2].set_title("images nearest each mixture mean (low | high)")
    fig.legend(handles=[Patch(color=PALETTE[0], label="low component"),
                        Patch(color=PALETTE[1], label="high component")],
               loc="lower center", ncol=2, frameon=False)
    bold_suptitle(fig, f"Modes of the PCA components of {args.dataset}")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig, RESULTS / f"pca_modes_{args.dataset}_pc{n}.png"


def plot_joint(args, train, shape, components, ratio, proj):
    ranked, chosen = choose_pairs(proj[:, :args.search_components], args.pairs)
    print(f"best pairs among the first {args.search_components} components (separation = mean term "
          f"of the Bhattacharyya distance; gain = over the better component alone; * = shown):")
    for gain, sep, (i, j), wmin in ranked[:10]:
        print(f"  {'*' if (i, j) in chosen else ' '} pc{i:<2} pc{j:<2} gain {gain:6.3f}  "
              f"separation {sep:6.3f}  minority cluster {100 * wmin:.0f}%")

    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.6 * n), gridspec_kw={"width_ratios": [2, 3, 7]},
                             squeeze=False)
    for row, (i, j) in enumerate(chosen):
        ax_comp, ax_hist, ax_ex = axes[row]
        z = proj[:, [i, j]]
        w, mu, cov = gmm2_2d(z)
        cluster = gmm_posterior_2d(z, w, mu, cov).argmax(1)
        show_components(ax_comp, [components[i], components[j]], shape)
        ax_comp.set_ylabel(f"pc{i} | pc{j}\n{100 * ratio[i]:.1f}% | {100 * ratio[j]:.1f}% var")

        lo, hi = np.percentile(z, 0.5, axis=0), np.percentile(z, 99.5, axis=0)
        ax_hist.hist2d(z[:, 0], z[:, 1], bins=150, range=[[lo[0], hi[0]], [lo[1], hi[1]]],
                       cmap="Greys", norm=LogNorm())
        for k in range(2):
            vals, vecs = np.linalg.eigh(cov[k])
            angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            ax_hist.add_patch(Ellipse(mu[k], 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=angle,
                                      fill=False, edgecolor=PALETTE[k], linewidth=2))
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        post = gmm_posterior_2d(grid, w, mu, cov)[:, 1].reshape(gx.shape)
        ax_hist.contour(gx, gy, post, levels=[0.5], colors="0.15", linewidths=1.2)
        ax_hist.set_xlim(lo[0], hi[0])
        ax_hist.set_ylim(lo[1], hi[1])
        ax_hist.set_xlabel(f"pc{i}")
        ax_hist.set_ylabel(f"pc{j}")
        ax_hist.grid(False)
        share = float(np.mean(cluster == 1))
        ax_hist.text(0.98, 0.95, f"separation {separation(mu, cov):.2f}\nsplit "
                     f"{100 * (1 - share):.0f}/{100 * share:.0f}", transform=ax_hist.transAxes,
                     ha="right", va="top", fontsize=9,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

        show_examples(ax_ex, train, [nearest(z, mu[0]), nearest(z, mu[1])], shape)
    axes[0, 0].set_title("the two components")
    axes[0, 1].set_title("train images in the pair's plane")
    axes[0, 2].set_title("images nearest each cluster mean")
    fig.legend(handles=[Patch(color=PALETTE[0], label="cluster A (2σ)"),
                        Patch(color=PALETTE[1], label="cluster B (2σ)")],
               loc="lower center", ncol=2, frameon=False)
    bold_suptitle(fig, f"Two-cluster modes in pairs of PCA components of {args.dataset}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig, RESULTS / f"pca_joint_modes_{args.dataset}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--view", choices=["bins", "modes", "joint"], default="bins")
    parser.add_argument("--components", type=int, default=6, help="bins / modes: rows")
    parser.add_argument("--bins", type=int, default=8, help="bins: bins per component (even)")
    parser.add_argument("--search_components", type=int, default=20,
                        help="joint: look for pairs among this many components")
    parser.add_argument("--pairs", type=int, default=3, help="joint: pairs to show")
    args = parser.parse_args()

    train, _ = get_dataset(args.dataset)
    shape = tuple(train[0][0].shape)                                  # (C, H, W)
    n_fit = args.search_components if args.view == "joint" else args.components
    mean, components, ratio = fit_pca(train, n_fit)
    proj = project(train, mean, components)

    set_style()
    view = {"bins": plot_bins, "modes": plot_modes, "joint": plot_joint}[args.view]
    fig, out = view(args, train, shape, components, ratio, proj)
    save(fig, out)


if __name__ == "__main__":
    main()
