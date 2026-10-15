"""Tests for data/cluster_factors.py, on synthetic images with two known categorical factors."""
import time

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import data.cluster_factors as cluster_factors
from data.cluster_factors import (balanced_renumbering, cluster_cache_path, kmeans, nmi,
                                  with_cluster_factors)
from data.download import DATA_ROOT, get_mnist

CORNERS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)


class Images(Dataset):
    """(N, 1, 4, 4) float images, all labelled 0."""

    def __init__(self, x):
        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], 0


def make_split(n, seed):
    """Factor A: square of side 20 on dims 0-1; factor B: square of side 8 on dims 2-3."""
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 4, n), rng.integers(0, 4, n)
    x = rng.normal(scale=0.5, size=(n, 16))
    x[:, 0:2] += 10 * CORNERS[a]
    x[:, 2:4] += 4 * CORNERS[b]
    return Images(x.reshape(n, 1, 4, 4)), a, b


@pytest.fixture(scope="module")
def fitted():
    train, a, b = make_split(2000, 0)
    test, ta, tb = make_split(500, 1)
    tr, te = with_cluster_factors(train, test, n_factors=2, n_clusters=4, n_dims=16)
    return tr, te, a, b, ta, tb


def test_factors_recover_the_planted_factors_and_are_independent(fitted):
    tr, _, a, b, _, _ = fitted
    assert nmi(tr.latent_idx[:, 0], a) > 0.9
    assert nmi(tr.latent_idx[:, 1], b) > 0.9
    assert nmi(tr.latent_idx[:, 0], tr.latent_idx[:, 1]) < 0.1
    assert tr.factor_nmi.shape == (2, 2) and tr.factor_nmi[0, 1] < 0.1
    assert tr.factor_names == ["clu0", "clu1"] and tr.factor_n_values == [4, 4]
    assert tr.explained_variance is None


def test_even_odd_split_is_balanced(fitted):
    tr, te, *_ = fitted
    for f in range(2):
        even = np.mean(tr.latent_idx[:, f] % 2 == 0)
        assert abs(even - 0.5) < 0.05
        assert tr.class_balance[f] == pytest.approx(even)
    assert np.array_equal(te.class_balance, tr.class_balance)
    assert tr.cluster_sizes.shape == (2, 4) and np.all(tr.cluster_sizes.sum(1) == 2000)
    assert np.array_equal(te.cluster_sizes, tr.cluster_sizes)


def test_test_split_uses_the_train_clusters(fitted):
    _, te, _, _, ta, tb = fitted
    assert te.latent_idx.shape == (500, 2)
    assert nmi(te.latent_idx[:, 0], ta) > 0.9
    assert nmi(te.latent_idx[:, 1], tb) > 0.9


def test_kmeans_is_deterministic_and_finds_the_blobs():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(100, 3)) + 20 * i for i in range(3)])
    c1, l1 = kmeans(x, 3, seed=5)
    c2, l2 = kmeans(x, 3, seed=5)
    assert np.array_equal(c1, c2) and np.array_equal(l1, l2)
    assert nmi(l1, np.repeat(np.arange(3), 100)) == pytest.approx(1.0)


def test_nmi_edge_cases():
    a = np.array([0, 0, 1, 1])
    assert nmi(a, a) == pytest.approx(1.0)
    assert nmi(a, np.array([0, 1, 0, 1])) == pytest.approx(0.0)
    assert nmi(a, np.zeros(4, dtype=int)) == 0.0


def test_balanced_renumbering_splits_fifty_fifty():
    labels = np.repeat(np.arange(4), [40, 30, 20, 10])
    new = balanced_renumbering(labels, 4)[labels]
    assert set(new[:40]) == {0}                           # the largest cluster opens group A
    assert np.mean(new % 2 == 0) == pytest.approx(0.5)
    assert sorted(set(new)) == [0, 1, 2, 3]


def test_balanced_renumbering_keeps_ids_below_c_when_sizes_are_lopsided():
    labels = np.repeat(np.arange(4), [50, 30, 10, 10])    # 50 | 50 would need 1 + 3 clusters
    assert sorted(balanced_renumbering(labels, 4)) == [0, 1, 2, 3]


@pytest.mark.parametrize("kwargs", [dict(n_factors=2, n_clusters=4, n_dims=6),    # needs 7
                                    dict(n_factors=1, n_clusters=1, n_dims=10)])
def test_invalid_arguments_raise_before_fitting(monkeypatch, kwargs):
    def fail(*a, **k):
        raise AssertionError("fit_pca must not be called")
    monkeypatch.setattr(cluster_factors, "fit_pca", fail)
    train, *_ = make_split(50, 0)
    with pytest.raises(ValueError):
        with_cluster_factors(train, train, **kwargs)


def test_cache_round_trip(tmp_path, monkeypatch):
    train, *_ = make_split(400, 0)
    test, *_ = make_split(100, 1)
    path = tmp_path / "sub" / "c.npz"
    tr, te = with_cluster_factors(train, test, 2, 4, n_dims=16, cache_path=path)
    assert path.exists()

    def fail(*a, **k):
        raise AssertionError("refit despite cache")
    monkeypatch.setattr(cluster_factors, "fit_pca", fail)
    tr2, te2 = with_cluster_factors(train, test, 2, 4, n_dims=16, cache_path=path)
    assert np.array_equal(tr.latent_idx, tr2.latent_idx) and np.array_equal(te.latent_idx, te2.latent_idx)
    assert np.array_equal(tr2.factor_nmi, tr.factor_nmi) and np.array_equal(te2.cluster_sizes, tr.cluster_sizes)
    assert np.array_equal(te2.class_balance, tr.class_balance)
    assert [p.name for p in path.parent.iterdir()] == ["c.npz"]


def test_cache_with_other_split_sizes_is_refitted(tmp_path):
    train, *_ = make_split(400, 0)
    path = tmp_path / "c.npz"
    with_cluster_factors(train, make_split(100, 1)[0], 2, 4, n_dims=16, cache_path=path)
    _, te = with_cluster_factors(train, make_split(120, 1)[0], 2, 4, n_dims=16, cache_path=path)
    assert te.latent_idx.shape[0] == 120


def test_cluster_cache_path(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_factors, "DATA_ROOT", tmp_path)
    assert cluster_cache_path("mnist", 6, 8, 50) == tmp_path / "clusters" / "mnist_f6_c8_d50.npz"


@pytest.mark.skipif(not (DATA_ROOT / "mnist" / "MNIST" / "raw").exists(), reason="MNIST not downloaded")
def test_mnist_factors_are_balanced():
    start = time.time()
    tr, te = with_cluster_factors(*get_mnist(), 6, 8)
    print(f"\nMNIST clustering: {time.time() - start:.1f}s")
    print("balance", np.round(tr.class_balance, 3))
    print("nmi\n", np.round(tr.factor_nmi, 3))
    assert tr.latent_idx.shape == (60000, 6) and te.latent_idx.shape == (10000, 6)
    for f in range(6):
        assert 0.45 < np.mean(tr.latent_idx[:, f] % 2 == 0) < 0.55
