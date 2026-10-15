"""
data/pca_factors.py — Shapes3D-style factors for any image dataset: the first n
principal components of the raw pixels, each cut into K equal-count bins.

    train, test = with_pca_factors(train, test, n_components=6, n_bins=8,
                                   cache_path=pca_cache_path("mnist", 6, 8))
    train.latent_idx                        # (N, 6) bin of every image on every component
    FeatureParityTask([2]).apply(train)     # task: parity of the bin on pc2

PCA is fitted on the train split as the network sees it (normalised, flattened);
the bin edges are the train quantiles, applied to both splits. With K = 2 the
parity task is the median split (linearly separable); larger K puts K - 1
parallel decision boundaries along the component.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.download import DATA_ROOT

CACHE_VERSION = 1       # bump when the fitting changes: older caches are then refitted


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


def pca_cache_path(dataset_kind, n_components, n_bins):
    """Where the PCA bins of a fixed-split dataset are cached."""
    return Path(DATA_ROOT) / "pca" / f"{dataset_kind}_pc{n_components}_bins{n_bins}.npz"


def with_pca_factors(train, test, n_components, n_bins, cache_path=None):
    """
    (train, test) wrapped in PCAFactorDataset: PCA fitted on `train`, each component
    cut at the train quantiles into `n_bins` bins. With `cache_path`, reuses the bins
    stored there when their version, settings and split sizes match, and stores them otherwise.
    """
    _check_bins(n_bins)
    cache_path = Path(cache_path) if cache_path is not None else None
    settings = dict(version=CACHE_VERSION, n_components=n_components, n_bins=n_bins)
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            files = set(cached.files)
            if (files >= {*settings, "train_idx", "test_idx", "explained_variance"}
                    and all(int(cached[k]) == v for k, v in settings.items())):
                train_idx, test_idx, ratio = (cached["train_idx"], cached["test_idx"],
                                              cached["explained_variance"])
                if len(train_idx) == len(train) and len(test_idx) == len(test):
                    return (PCAFactorDataset(train, train_idx, ratio, n_bins),
                            PCAFactorDataset(test, test_idx, ratio, n_bins))

    with torch.random.fork_rng(devices=[]):         # the DataLoaders must not advance the caller's RNG
        mean, components, ratio = fit_pca(train, n_components)
        train_proj = project(train, mean, components)
        edges = bin_edges(train_proj, n_bins)
        train_idx = assign_bins(train_proj, edges)
        test_idx = assign_bins(project(test, mean, components), edges)

    if cache_path is not None:
        # parallel jobs may fill the same cache: write a private file, then rename it
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, **settings, mean=mean, components=components, explained_variance=ratio,
                 edges=edges, train_idx=train_idx, test_idx=test_idx)
        os.replace(tmp, cache_path)
    return (PCAFactorDataset(train, train_idx, ratio, n_bins),
            PCAFactorDataset(test, test_idx, ratio, n_bins))
