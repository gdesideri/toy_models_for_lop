"""
tasks/permuted_pixels.py — Permuted-pixel tasks on image datasets.

A task is a permutation of the pixel grid in which a `perm_percentage` fraction
of the positions is shuffled among itself (0 → identity, 1 → fully permuted).
The same permutation is applied to every image, and to BOTH splits: the test
set must carry the task the model was trained on.
"""
from __future__ import annotations

import numpy as np
import torch


def sample_pixel_permutation(n_pixels, perm_percentage, rng):
    """Samples one task: a permutation of `n_pixels` positions, `perm_percentage` of them moved."""
    perm = np.arange(n_pixels)
    n_perm = int(round(n_pixels * perm_percentage))
    if n_perm > 1:
        idx = rng.choice(n_pixels, n_perm, replace=False)
        perm[idx] = idx[rng.permutation(n_perm)]
    return perm


class PermutedPixels(torch.utils.data.Dataset):
    """Wraps a dataset and re-orders the pixels of every image with `perm`."""

    def __init__(self, dataset, perm):
        self.dataset = dataset
        self.perm = torch.from_numpy(np.asarray(perm)).long()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        shape = x.shape
        x = x.reshape(-1, self.perm.numel())[:, self.perm].reshape(shape)
        return x, y


def permute_dataset(dataset, perm):
    """Applies the sampled task to a dataset (no-op for the identity permutation)."""
    if np.array_equal(perm, np.arange(len(perm))):
        return dataset
    return PermutedPixels(dataset, perm)
