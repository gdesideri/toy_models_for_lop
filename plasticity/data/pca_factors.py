"""
data/pca_factors.py — Shapes3D-style factors for any image dataset: the first n
principal components of the raw pixels; every image is scored on every component and
each score is cut into K equal-count bins.

    train, test = with_pca_factors(train, test, n_components=6, n_bins=8,
                                   cache_path=pca_cache_path("mnist", 6, 8))
    train.latent_idx                        # (N, 6) bin of every image on every component
    FeatureParityTask([2]).apply(train)     # task: parity of the bin on pc2
    FeatureUpperHalfTask(2).apply(train)    # task: bin in the upper half (score above the train median)

PCA is fitted on the train split as the network sees it (normalised, flattened);
the bin edges are the train quantiles, applied to both splits. The score of image x
on component v (unit length) is
    projection (default)  (x - mean) . v
    cosine                (x - mean) . v / |x - mean|: the cosine similarity of the centred
                          image with the component, in [-1, 1]; the direction in which the
                          image differs from the mean image, whatever its distance from it
With K = 2 the parity task is the median split; larger K puts K - 1 boundaries along the
component (parallel planes for projections, cones around it for cosines). The upper-half
task is the median split for every even K.

Build a benchmark, cache it where experiments/run_tasks_stream.py finds it, and print
what it is like (label balance under both rules, overlap between components and with the class):
    uv run python -m data.pca_factors --dataset cifar10 --components 6 --bins 10 --score cosine
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.download import DATA_ROOT

CACHE_VERSION = 1       # bump when the fitting changes: older caches are then refitted
SCORES = ("projection", "cosine")
LEGACY_SETTINGS = {"score": "projection"}    # what a cache written before a setting existed used


def _batches(dataset, batch_size=1024):
    """The flattened images of `dataset` in float64, a batch at a time."""
    for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        yield x.reshape(x.shape[0], -1).double()


def fit_pca(dataset, n_components):
    """
    (mean (D,), components (n, D), explained-variance ratio (n,)) of the flattened
    images of `dataset`. The rows of `components` are orthonormal, by decreasing
    variance, each signed so that its largest-magnitude entry is positive.
    """
    count, sum_x, sum_xx = 0, 0.0, 0.0
    for x in _batches(dataset):
        count += x.shape[0]
        sum_x = sum_x + x.sum(0)
        sum_xx = sum_xx + x.T @ x
    mean = sum_x / count
    cov = sum_xx / count - torch.outer(mean, mean)
    if not 1 <= n_components <= cov.shape[0]:
        raise ValueError(f"n_components must be 1-{cov.shape[0]} (the number of pixels), got {n_components}")

    eigvals, eigvecs = torch.linalg.eigh(cov)                       # ascending
    order = torch.argsort(eigvals, descending=True)[:n_components]
    components = eigvecs[:, order].T
    rows = torch.arange(n_components)
    components = components * torch.sign(components[rows, components.abs().argmax(1)])[:, None]
    ratio = eigvals[order] / eigvals.clamp(min=0).sum()
    return mean.numpy(), components.numpy(), ratio.numpy()


def project(dataset, mean, components):
    """(N, n) coordinates of the images of `dataset` on `components`."""
    mean, components = torch.from_numpy(mean), torch.from_numpy(components)
    return torch.cat([(x - mean) @ components.T for x in _batches(dataset)]).numpy()


def cosine(dataset, mean, components):
    """(N, n) cosine similarity of every centred image (x - mean) with every component: its
    projection divided by |x - mean|, in [-1, 1]; 0 for an image equal to the mean."""
    mean, components = torch.from_numpy(mean), torch.from_numpy(components)
    out = []
    for x in _batches(dataset):
        d = x - mean
        norm = d.norm(dim=1, keepdim=True)
        out.append((d @ components.T) / torch.where(norm > 0, norm, torch.ones_like(norm)))
    return torch.cat(out).numpy()


def scores(dataset, mean, components, score="projection"):
    """(N, n) score of every image of `dataset` on every component: `projection` or `cosine`."""
    _check_score(score)
    return (project if score == "projection" else cosine)(dataset, mean, components)


def _check_score(score):
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")


def _check_bins(n_bins):
    if n_bins < 2 or n_bins % 2:
        raise ValueError(f"n_bins must be even and >= 2 (the parity of an odd number of bins "
                         f"is unbalanced), got {n_bins}")


def bin_edges(train_proj, n_bins):
    """(n, K - 1) inner quantile edges of every component: K bins of equal train counts."""
    _check_bins(n_bins)
    return np.quantile(train_proj, np.arange(1, n_bins) / n_bins, axis=0).T


def assign_bins(proj, edges):
    """(N, n) bin index 0..K-1 of every coordinate of `proj`."""
    return np.stack([np.searchsorted(edges[j], proj[:, j], side="right")
                     for j in range(proj.shape[1])], axis=1).astype(np.int64)


class PCAFactorDataset(Dataset):
    """`dataset` unchanged, plus its PCA bins as Shapes3D-style latent factors."""

    def __init__(self, dataset, latent_idx, explained_variance, n_bins, factor_names=None):
        self.dataset = dataset
        self.latent_idx = latent_idx
        self.explained_variance = explained_variance
        self.factor_names = factor_names or [f"pc{j}" for j in range(latent_idx.shape[1])]
        self.factor_n_values = [n_bins] * latent_idx.shape[1]
        self.classes = getattr(dataset, "classes", None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


def pca_cache_path(dataset_kind, n_components, n_bins, score="projection"):
    """Where the PCA bins of a fixed-split dataset are cached (one file per score)."""
    _check_score(score)
    suffix = "" if score == "projection" else "_cos"
    return Path(DATA_ROOT) / "pca" / f"{dataset_kind}_pc{n_components}_bins{n_bins}{suffix}.npz"


def with_pca_factors(train, test, n_components, n_bins, cache_path=None, score="projection"):
    """
    (train, test) wrapped in PCAFactorDataset: PCA fitted on `train`, every image scored on
    every component (`score`: projection or cosine) and each score cut at the train quantiles
    into `n_bins` bins. With `cache_path`, reuses the bins stored there when their version,
    settings and split sizes match, and stores them otherwise.
    """
    _check_bins(n_bins)
    _check_score(score)
    cache_path = Path(cache_path) if cache_path is not None else None
    settings = dict(version=CACHE_VERSION, n_components=n_components, n_bins=n_bins, score=score)
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            stored = {**LEGACY_SETTINGS, **{k: cached[k].item() for k in settings if k in cached.files}}
            if (set(cached.files) >= {"train_idx", "test_idx", "explained_variance"}
                    and all(stored.get(k) == v for k, v in settings.items())):
                train_idx, test_idx, ratio = (cached["train_idx"], cached["test_idx"],
                                              cached["explained_variance"])
                if len(train_idx) == len(train) and len(test_idx) == len(test):
                    return (PCAFactorDataset(train, train_idx, ratio, n_bins),
                            PCAFactorDataset(test, test_idx, ratio, n_bins))

    with torch.random.fork_rng(devices=[]):         # the DataLoaders must not advance the caller's RNG
        mean, components, ratio = fit_pca(train, n_components)
        train_scores = scores(train, mean, components, score)
        edges = bin_edges(train_scores, n_bins)
        train_idx = assign_bins(train_scores, edges)
        test_idx = assign_bins(scores(test, mean, components, score), edges)

    if cache_path is not None:
        # parallel jobs may fill the same cache: write a private file, then rename it
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, **settings, mean=mean, components=components, explained_variance=ratio,
                 edges=edges, train_idx=train_idx, test_idx=test_idx)
        os.replace(tmp, cache_path)
    return (PCAFactorDataset(train, train_idx, ratio, n_bins),
            PCAFactorDataset(test, test_idx, ratio, n_bins))


# ── building and describing a benchmark ──────────────────────────────────────

LABEL_RULES = {                              # a task's label from the bin, as the task classes make it
    "parity": lambda idx, n_bins: idx % 2 == 0,                  # FeatureParityTask
    "upper_half": lambda idx, n_bins: idx >= n_bins / 2,         # FeatureUpperHalfTask
}


def _true_labels(dataset):
    """What the factors can be compared with: the class, or Shapes3D's own six factors."""
    if hasattr(dataset, "targets"):
        return {"the class": np.asarray(dataset.targets)}
    if hasattr(dataset, "latent_idx"):
        from data.download import SHAPES3D_FACTOR_NAMES
        latent = np.asarray(dataset.latent_idx)
        return {name: latent[:, f] for f, name in enumerate(SHAPES3D_FACTOR_NAMES)}
    return {}


def main():
    import argparse

    from data.cluster_factors import nmi
    from data.factory import build_dataset

    parser = argparse.ArgumentParser(description="Build the PCA factors of a dataset (cached where "
                                                 "run_tasks_stream finds them) and print what they are like.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--score", choices=SCORES, default="projection")
    parser.add_argument("--n_samples", type=int, default=None, help="shapes3d only")
    parser.add_argument("--seed", type=int, default=0, help="shapes3d only: the seed of the split")
    a = parser.parse_args()

    base_train, base_test = build_dataset(a.dataset, n_samples=a.n_samples, rng=np.random.default_rng(a.seed))
    # as in run_tasks_stream: Shapes3D's split depends on the seed, so it is not cached
    cache = None if a.dataset == "shapes3d" else pca_cache_path(a.dataset, a.components, a.bins, a.score)
    train, test = with_pca_factors(base_train, base_test, a.components, a.bins, cache_path=cache, score=a.score)
    n, k, w = a.components, a.bins, 16
    names, idx = train.factor_names, train.latent_idx

    print(f"{a.dataset}: {a.score} scores on {n} components (" +
          ", ".join(f"{100 * r:.1f}%" for r in train.explained_variance) +
          f" of the pixel variance), {k} equal-count bins" + (f"; cached in {cache}" if cache else ""))
    print(f"\n{'label 1, train / test':22s}" + "".join(f"{s:>{w}s}" for s in names))
    for rule, label in LABEL_RULES.items():
        print(f"  {rule:20s}" + "".join(
            f"{label(idx[:, j], k).mean():.3f} / {label(test.latent_idx[:, j], k).mean():.3f}".rjust(w)
            for j in range(n)))
    print(f"\n{'same label, next task':22s}" + "".join(f"{names[j]}>{names[(j + 1) % n]}".rjust(w) for j in range(n)))
    for rule, label in LABEL_RULES.items():
        lab = label(idx, k)
        print(f"  {rule:20s}" + "".join(f"{(lab[:, j] == lab[:, (j + 1) % n]).mean():.1%}".rjust(w)
                                        for j in range(n)))
    if n > 1:
        overlap = np.array([[nmi(idx[:, i], idx[:, j]) for j in range(n)] for i in range(n)])[~np.eye(n, dtype=bool)]
        print(f"\nNMI between the components' bins (0 = unrelated, 1 = the same): "
              f"max {overlap.max():.3f}, mean {overlap.mean():.3f}")
    for what, y in _true_labels(base_train).items():
        print(f"NMI with {what}".ljust(22) + "".join(f"{nmi(idx[:, j], y):.3f}".rjust(w) for j in range(n)))


if __name__ == "__main__":
    main()
