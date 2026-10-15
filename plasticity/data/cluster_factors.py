"""
data/cluster_factors.py — Shapes3D-style categorical factors for any image dataset:
F clusterings of the images with C clusters each, made non-redundant by orthogonalisation.

    train, test = with_cluster_factors(train, test, n_factors=6, n_clusters=8,
                                       cache_path=cluster_cache_path("mnist", 6, 8, 50))
    train.latent_idx                          # (N, 6) cluster id of every image on every factor
    FeatureParityTask([2]).apply(train)       # task: even vs odd cluster id of factor 2

The images are projected on their top n_dims pixel principal components. Factor f is a k-means
clustering of that space; before factor f + 1 the subspace spanned by the centroids of factor f
is projected out, so later factors must find structure the earlier ones did not (Cui, Fern & Dy
2007, "Non-redundant multi-view clustering via orthogonalization"). The clusters of every factor
are renumbered so that even ids and odd ids each hold about half of the train images: the
parity task is then a balanced "cluster group A vs group B" task.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from data.download import DATA_ROOT
from data.pca_factors import PCAFactorDataset, fit_pca, project


def _sq_dists(x, centroids):
    """(N, k) squared distances from every row of x to every centroid."""
    d = (x ** 2).sum(1)[:, None] - 2 * x @ centroids.T + (centroids ** 2).sum(1)[None]
    return np.maximum(d, 0)


def _kmeans_once(x, k, iters, rng):
    # k-means++ seeding
    centroids = [x[rng.integers(len(x))]]
    closest = _sq_dists(x, centroids[0][None])[:, 0]
    for _ in range(1, k):
        total = closest.sum()
        p = closest / total if total > 0 else None
        centroids.append(x[rng.choice(len(x), p=p)])
        closest = np.minimum(closest, _sq_dists(x, centroids[-1][None])[:, 0])
    centroids = np.stack(centroids)

    labels = None
    for _ in range(iters):
        d = _sq_dists(x, centroids)
        new = d.argmin(1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = labels == j
            if members.any():
                centroids[j] = x[members].mean(0)
            else:                                   # empty: re-seed at the worst-fitted point
                far = d[np.arange(len(x)), labels].argmax()
                centroids[j], labels[far] = x[far], j
                d[far, labels[far]] = 0
    d = _sq_dists(x, centroids)
    labels = d.argmin(1)
    return centroids, labels, d[np.arange(len(x)), labels].sum()


def kmeans(x, k, n_init=4, iters=100, seed=0):
    """(centroids (k, D), labels (N,)) of the best of `n_init` k-means++ / Lloyd runs (lowest inertia)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    runs = [_kmeans_once(x, k, iters, rng) for _ in range(n_init)]
    centroids, labels, _ = min(runs, key=lambda r: r[2])
    return centroids, labels


def nmi(a, b):
    """Normalised mutual information (arithmetic-mean normalisation) of two label arrays; 0 if either is constant."""
    a, b = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    ha, hb = -(pa * np.log(pa)).sum(), -(pb * np.log(pb)).sum()
    if ha == 0 or hb == 0:
        return 0.0
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])).sum()
    return float(mi / ((ha + hb) / 2))


def _check_sizes(n_factors, n_clusters, n_dims):
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
    need = n_factors * (n_clusters - 1) + 1
    if n_dims < need:
        raise ValueError(f"{n_factors} factors of {n_clusters} clusters need at least {need} "
                         f"dimensions (each factor removes up to {n_clusters - 1}), got {n_dims}")


def fit_clusterings(z, n_factors, n_clusters, seed=0):
    """
    [(centroids (C, D), basis (D, r))] * n_factors. Factor f is a k-means clustering of z_f;
    `basis` spans the centroids' differences (r <= C - 1) and is projected out to get z_{f+1}.
    """
    _check_sizes(n_factors, n_clusters, z.shape[1])
    z, out = np.asarray(z, dtype=np.float64), []
    for f in range(n_factors):
        centroids, labels = kmeans(z, n_clusters, seed=seed + f)
        sizes = np.bincount(labels, minlength=n_clusters)
        spread = centroids - (sizes[:, None] * centroids).sum(0) / sizes.sum()
        _, s, vt = np.linalg.svd(spread, full_matrices=False)
        basis = vt[s > 1e-8 * s.max()].T
        out.append((centroids, basis))
        z = z - (z @ basis) @ basis.T
    return out


def assign_clusterings(z, clusterings):
    """(N, F) int64 nearest-centroid ids of `z`, replaying the orthogonalisation chain."""
    z, cols = np.asarray(z, dtype=np.float64), []
    for centroids, basis in clusterings:
        cols.append(_sq_dists(z, centroids).argmin(1))
        z = z - (z @ basis) @ basis.T
    return np.stack(cols, axis=1).astype(np.int64)


def balanced_renumbering(labels, n_clusters):
    """
    Array mapping old cluster id -> new id, a permutation of 0..C-1. Clusters, largest first,
    join whichever of two groups holds fewer images (a group that already has its share of the
    ids, ceil(C/2) even or floor(C/2) odd, is skipped, so no id exceeds C - 1); group A's
    clusters become 0, 2, 4, ... and group B's 1, 3, 5, ..., largest first.
    """
    sizes = np.bincount(labels, minlength=n_clusters)
    capacity = [(n_clusters + 1) // 2, n_clusters // 2]
    groups, load = ([], []), [0, 0]
    for c in np.argsort(-sizes, kind="stable"):
        open_groups = [g for g in (0, 1) if len(groups[g]) < capacity[g]]
        g = min(open_groups, key=lambda g: load[g])
        groups[g].append(c)
        load[g] += sizes[c]
    new = np.empty(n_clusters, dtype=np.int64)
    for g in (0, 1):
        for rank, c in enumerate(groups[g]):
            new[c] = 2 * rank + g
    return new


def cluster_cache_path(dataset_kind, n_factors, n_clusters, n_dims):
    """Where the cluster factors of a fixed-split dataset are cached."""
    return Path(DATA_ROOT) / "clusters" / f"{dataset_kind}_f{n_factors}_c{n_clusters}_d{n_dims}.npz"


def _wrap(dataset, idx, n_clusters, stats):
    wrapped = PCAFactorDataset(dataset, idx, None, n_clusters,
                               factor_names=[f"clu{f}" for f in range(idx.shape[1])])
    wrapped.factor_nmi, wrapped.class_balance, wrapped.cluster_sizes = stats
    return wrapped


def with_cluster_factors(train, test, n_factors, n_clusters, n_dims=50, cache_path=None):
    """
    (train, test) wrapped in PCAFactorDataset with `n_factors` cluster factors of `n_clusters`
    values (ids renumbered for an even/odd balanced split of train). PCA and clustering are fitted
    on `train` with a fixed seed. Both wrappers also carry `factor_nmi` (F, F), `class_balance`
    (F,) (train fraction of even ids) and `cluster_sizes` (F, C) (train counts). With
    `cache_path`, reuses the factors stored there when their split sizes match.
    """
    _check_sizes(n_factors, n_clusters, n_dims)
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["train_idx"]) == len(train) and len(cached["test_idx"]) == len(test):
            stats = (cached["factor_nmi"], cached["class_balance"], cached["cluster_sizes"])
            return (_wrap(train, cached["train_idx"], n_clusters, stats),
                    _wrap(test, cached["test_idx"], n_clusters, stats))

    mean, components, _ = fit_pca(train, n_dims)
    train_z, test_z = project(train, mean, components), project(test, mean, components)
    clusterings = fit_clusterings(train_z, n_factors, n_clusters, seed=0)
    train_idx, test_idx = assign_clusterings(train_z, clusterings), assign_clusterings(test_z, clusterings)
    for f in range(n_factors):
        mapping = balanced_renumbering(train_idx[:, f], n_clusters)
        train_idx[:, f], test_idx[:, f] = mapping[train_idx[:, f]], mapping[test_idx[:, f]]

    factor_nmi = np.array([[nmi(train_idx[:, i], train_idx[:, j]) for j in range(n_factors)]
                           for i in range(n_factors)])
    class_balance = (train_idx % 2 == 0).mean(0)
    cluster_sizes = np.stack([np.bincount(train_idx[:, f], minlength=n_clusters) for f in range(n_factors)])
    stats = (factor_nmi, class_balance, cluster_sizes)

    if cache_path is not None:
        # parallel jobs may fill the same cache: write a private file, then rename it
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, train_idx=train_idx, test_idx=test_idx, factor_nmi=factor_nmi,
                 class_balance=class_balance, cluster_sizes=cluster_sizes)
        os.replace(tmp, cache_path)
    return _wrap(train, train_idx, n_clusters, stats), _wrap(test, test_idx, n_clusters, stats)
