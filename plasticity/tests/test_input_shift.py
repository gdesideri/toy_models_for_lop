"""data/input_shift.py: input statistics, centring and offsets, and the attribute pass-through."""
import copy

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from data.input_shift import ShiftedInputs, input_stats, mean_image, offset_for_ratio, shift_for


class Images(Dataset):
    """(N, 1, 2, 2) float images with label 0, plus attributes a factor dataset would carry."""

    def __init__(self, x):
        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.latent_idx = np.arange(len(self.x))[:, None]
        self.targets = [0] * len(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], 0


def random_images(n=500, seed=0, mean=2.0):
    return Images(np.random.default_rng(seed).normal(size=(n, 1, 2, 2)) + mean)


def test_input_stats_on_known_images():
    # two images [1, 1] and [1, -1]: mean image [1, 0], per-input variances 0 and 1
    stats = input_stats(Images([[[[1.0, 1.0]]], [[[1.0, -1.0]]]]))
    assert stats["shared_push"] == pytest.approx(1.0)
    assert stats["mean_energy"] == pytest.approx(2.0)
    assert stats["shared_fraction"] == pytest.approx(0.5)
    assert stats["unit_spread"] == pytest.approx(1.0)       # sqrt(2 * mean variance 0.5)
    assert stats["push_over_spread"] == pytest.approx(1.0)
    assert stats["n_inputs"] == 2 and stats["n_images"] == 2


def test_centring_removes_the_shared_push_and_keeps_the_spread():
    data = random_images()
    before = input_stats(data)
    after = input_stats(ShiftedInputs(data, shift_for(data, center=1.0, offset=0.0)))
    assert before["shared_push"] > 10
    assert after["shared_push"] < 1e-8
    assert after["unit_spread"] == pytest.approx(before["unit_spread"], rel=1e-6)


def test_partial_centring_scales_the_shared_push():
    data = random_images()
    m2 = input_stats(data)["shared_push"]
    half = input_stats(ShiftedInputs(data, shift_for(data, center=0.5, offset=0.0)))["shared_push"]
    assert half == pytest.approx(0.25 * m2, rel=1e-5)


def test_offset_adds_its_energy():
    data = random_images(mean=0.0)
    s = input_stats(data)
    c = 0.7
    shifted = input_stats(ShiftedInputs(data, shift_for(data, center=0.0, offset=c)))
    expected = s["shared_push"] + 2 * c * s["mean_total"] + c ** 2 * s["n_inputs"]
    assert shifted["shared_push"] == pytest.approx(expected, rel=1e-5)


def test_offset_for_ratio_reaches_the_target():
    data = random_images(mean=0.0)
    c = offset_for_ratio(input_stats(data), 50.0)
    shifted = input_stats(ShiftedInputs(data, shift_for(data, center=0.0, offset=c)))
    assert shifted["push_over_spread"] == pytest.approx(50.0, rel=1e-4)


def test_no_shift_when_both_are_zero():
    assert shift_for(random_images(), center=0.0, offset=0.0) is None


def test_mean_image_has_the_image_shape():
    data = random_images()
    m = mean_image(data)
    assert m.shape == (1, 2, 2)
    np.testing.assert_allclose(m.numpy(), data.x.mean(0).numpy(), atol=1e-5)


def test_shifted_inputs_shift_images_and_pass_attributes_through():
    data = random_images(n=10)
    shift = torch.full((1, 2, 2), 3.0)
    shifted = ShiftedInputs(data, shift)
    x, y = shifted[4]
    assert torch.allclose(x, data[4][0] + 3.0) and y == 0
    assert len(shifted) == 10
    assert shifted.latent_idx is data.latent_idx and shifted.targets is data.targets
    with pytest.raises(AttributeError):
        shifted.not_an_attribute
    copy.deepcopy(shifted)                                  # no recursion through __getattr__


def test_statistics_leave_the_torch_rng_untouched():
    data = random_images()
    torch.manual_seed(0)
    input_stats(data)
    mean_image(data)
    after = torch.rand(1)
    torch.manual_seed(0)
    assert torch.equal(after, torch.rand(1))
