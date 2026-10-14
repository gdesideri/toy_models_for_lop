"""FeatureParityTask: unchanged on Shapes3D, usable on any number of factors."""
import numpy as np
import pytest
import torch

from tasks.feature_streams import FeatureParityTask


class Latents:
    """The minimum FeatureParityTask.apply needs: images and a latent_idx array."""

    def __init__(self, latent_idx):
        self.latent_idx = np.asarray(latent_idx)

    def __len__(self):
        return len(self.latent_idx)

    def __getitem__(self, i):
        return torch.zeros(1), 0


def test_shapes3d_task_unchanged():
    task = FeatureParityTask([5])
    assert task.name == "parity(object_azimuth)"
    latent = np.zeros((15, 6), dtype=np.int64)
    latent[:, 5] = np.arange(15)
    np.testing.assert_array_equal(task.labels(latent), (np.arange(15) % 2 == 0).astype(np.int64))
    labelled = task.apply(Latents(latent))
    assert labelled[2][1] == 1 and labelled[3][1] == 0


def test_factor_beyond_the_six_shapes3d_ones():
    latent = np.zeros((4, 8), dtype=np.int64)
    latent[:, 7] = [0, 1, 2, 3]
    task = FeatureParityTask([7])
    assert task.name == "parity(7)"
    labelled = task.apply(Latents(latent))
    assert [labelled[i][1] for i in range(4)] == [1, 0, 1, 0]


def test_factor_missing_from_the_dataset_rejected_at_apply():
    with pytest.raises(ValueError, match="factors 0-5"):
        FeatureParityTask([6]).apply(Latents(np.zeros((4, 6), dtype=np.int64)))


@pytest.mark.parametrize("factors", [[], [1, 1], [-1]])
def test_invalid_factor_lists_rejected(factors):
    with pytest.raises(ValueError):
        FeatureParityTask(factors)
