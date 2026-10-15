"""Tests for data/pca_factors.py, on synthetic images with known principal axes."""
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import data.pca_factors as pca_factors
from data.download import DATA_ROOT
from data.pca_factors import fit_pca, pca_cache_path, with_pca_factors

# pixel j has standard deviation SCALES[j]: the principal axes are pixels 0, 1, 2, ... in order
SCALES = np.array([5.0, 3.0, 2.0, 1.0] + [0.1] * 12)


class Images(Dataset):
    """(N, 1, 4, 4) float images, all labelled 0."""

    def __init__(self, x):
        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], 0


def make_split(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 16)) * SCALES
    z[:, 0] += shift
    return Images(z.reshape(n, 1, 4, 4))


def test_fit_pca_recovers_the_principal_axes():
    mean, components, ratio = fit_pca(make_split(4000, 0), n_components=3)
    assert mean.shape == (16,) and components.shape == (3, 16) and ratio.shape == (3,)
    for j in range(3):
        assert components[j, j] > 0.99          # axis j, signed so its largest entry is positive
    assert np.all(np.diff(ratio) < 0)
    np.testing.assert_allclose(ratio, SCALES[:3] ** 2 / (SCALES ** 2).sum(), rtol=0.1)


def test_components_are_orthonormal():
    _, components, _ = fit_pca(make_split(2000, 1), n_components=5)
    np.testing.assert_allclose(components @ components.T, np.eye(5), atol=1e-8)


def test_too_many_components_rejected():
    with pytest.raises(ValueError, match="n_components"):
        fit_pca(make_split(100, 0), n_components=17)


@pytest.mark.parametrize("n_bins", [2, 4, 8, 16])
def test_train_bins_and_parity_are_balanced(n_bins):
    train, _ = with_pca_factors(make_split(4000, 0), make_split(1000, 1), 3, n_bins)
    assert train.latent_idx.shape == (4000, 3)
    for j in range(3):
        counts = np.bincount(train.latent_idx[:, j], minlength=n_bins)
        assert len(counts) == n_bins
        assert counts.max() - counts.min() <= 1
    np.testing.assert_allclose((train.latent_idx % 2 == 0).mean(axis=0), 0.5, atol=0.01)


def test_test_split_is_binned_with_the_train_edges():
    # the test images sit far out along pc0, so all of them land in the top train bin
    _, test = with_pca_factors(make_split(4000, 0), make_split(500, 1, shift=100.0), 3, 8)
    assert np.all(test.latent_idx[:, 0] == 7)


def test_wrapper_keeps_images_and_adds_factors():
    base = make_split(1000, 0)
    train, _ = with_pca_factors(base, make_split(200, 1), 3, 4)
    x, y = train[5]
    assert torch.equal(x, base[5][0]) and y == 0
    assert len(train) == 1000
    assert train.factor_names == ["pc0", "pc1", "pc2"]
    assert train.factor_n_values == [4, 4, 4]
    assert train.explained_variance.shape == (3,)


@pytest.mark.parametrize("n_bins", [0, 1, 3, 7])
def test_odd_or_too_few_bins_rejected_before_fitting(n_bins, monkeypatch):
    monkeypatch.setattr(pca_factors, "fit_pca", lambda *a, **k: pytest.fail("fitted before checking n_bins"))
    with pytest.raises(ValueError, match="n_bins"):
        with_pca_factors(make_split(100, 0), make_split(50, 1), 3, n_bins)


def test_cache_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "pca" / "toy_pc3_bins8.npz"
    train, test = with_pca_factors(make_split(1000, 0), make_split(200, 1), 3, 8, cache_path=path)
    assert list(path.parent.iterdir()) == [path]          # written atomically: no temp file left
    monkeypatch.setattr(pca_factors, "fit_pca", lambda *a, **k: pytest.fail("refitted despite the cache"))
    train2, test2 = with_pca_factors(make_split(1000, 0), make_split(200, 1), 3, 8, cache_path=path)
    np.testing.assert_array_equal(train2.latent_idx, train.latent_idx)
    np.testing.assert_array_equal(test2.latent_idx, test.latent_idx)
    np.testing.assert_array_equal(train2.explained_variance, train.explained_variance)


def test_cache_for_a_different_split_size_is_recomputed(tmp_path):
    path = tmp_path / "toy.npz"
    with_pca_factors(make_split(1000, 0), make_split(200, 1), 3, 8, cache_path=path)
    train, test = with_pca_factors(make_split(800, 0), make_split(100, 1), 3, 8, cache_path=path)
    assert train.latent_idx.shape == (800, 3) and test.latent_idx.shape == (100, 3)


def test_cache_path_names_every_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(pca_factors, "DATA_ROOT", tmp_path)
    assert pca_cache_path("mnist", 6, 8) == tmp_path / "pca" / "mnist_pc6_bins8.npz"
    paths = {pca_cache_path(d, n, k) for d in ("mnist", "cifar10") for n in (3, 6) for k in (2, 8)}
    assert len(paths) == 8


@pytest.mark.skipif(not (DATA_ROOT / "mnist" / "MNIST" / "raw").exists(), reason="MNIST not downloaded")
def test_mnist_end_to_end():
    from data.download import get_mnist
    train, test = with_pca_factors(*get_mnist(), 6, 8)
    assert train.latent_idx.shape == (60000, 6) and test.latent_idx.shape == (10000, 6)
    assert np.all(np.diff(train.explained_variance) < 0)
    # test bins come from the train edges, so test parity is only close to balanced
    assert np.all(np.abs((test.latent_idx % 2 == 0).mean(axis=0) - 0.5) < 0.05)


def test_cache_hit_and_miss_leave_the_torch_rng_in_the_same_state(tmp_path):
    path = tmp_path / "toy.npz"
    args = (make_split(1000, 0), make_split(200, 1), 3, 8)
    torch.manual_seed(0)
    with_pca_factors(*args, cache_path=path)               # miss: fits and projects
    after_miss = torch.rand(1)
    torch.manual_seed(0)
    with_pca_factors(*args, cache_path=path)               # hit
    assert torch.equal(torch.rand(1), after_miss)


@pytest.mark.parametrize("stale", [{"version": -1}, {"n_bins": 4}, {"n_components": 2}, {"legacy": True}])
def test_cache_with_another_version_or_settings_is_refitted(tmp_path, stale):
    path = tmp_path / "toy.npz"
    train, test = with_pca_factors(make_split(1000, 0), make_split(200, 1), 3, 8, cache_path=path)
    fields = dict(version=pca_factors.CACHE_VERSION, n_components=3, n_bins=8,
                  explained_variance=np.zeros(3), train_idx=np.zeros((1000, 3), dtype=np.int64),
                  test_idx=np.zeros((200, 3), dtype=np.int64))
    if stale.pop("legacy", False):
        del fields["version"], fields["n_components"], fields["n_bins"]
    np.savez(path, **{**fields, **stale})
    train2, test2 = with_pca_factors(make_split(1000, 0), make_split(200, 1), 3, 8, cache_path=path)
    np.testing.assert_array_equal(train2.latent_idx, train.latent_idx)
    with np.load(path) as cached:
        assert cached["version"] == pca_factors.CACHE_VERSION
