"""
data/factory.py — Build a (train, test) split by name and describe its shape.

    train, test = build_dataset("shapes3d", n_samples=20_000, target_factor=5, rng=rng)
    (c, h, w), n_classes = dataset_spec(train)
    train_loader, test_loader = build_loaders(train, test, batch_size=256, num_workers=4)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from data.download import get_dataset, SHAPES3D_FACTOR_N_VALUES


def binary_target(factor_indices, num_factor_values):
    """
    Binary label of an ordered factor: 1 for the upper half of its K values,
    y = int(index >= K / 2), so K=10: 0-4 -> 0, 5-9 -> 1 and K=15: 0-7 -> 0, 8-14 -> 1.

    Works elementwise on numpy arrays and torch tensors; `num_factor_values` may be
    an int or one K per column, e.g. binary_target(latent_idx, SHAPES3D_FACTOR_N_VALUES)
    labels an (N, 6) array by all six factors at once.
    """
    import torch

    if torch.is_tensor(factor_indices):
        k = torch.as_tensor(num_factor_values, dtype=torch.float64, device=factor_indices.device)
        return (factor_indices >= k / 2.0).long()
    return (np.asarray(factor_indices) >= np.asarray(num_factor_values) / 2.0).astype(np.int64)


# Named label rules for Shapes3D: factor-index -> class maps, by number of values.
LABEL_RULES = {
    # even factor index -> class 1, odd -> class 0 (object_azimuth: 8 vs 7 bins)
    "parity": lambda n_values: [1 - i % 2 for i in range(n_values)],
    # upper half of the ordered values -> class 1 (binary_target)
    "upper_half": lambda n_values: binary_target(np.arange(n_values), n_values).tolist(),
}


def build_dataset(
    kind: str,
    n_samples: Optional[int] = None,
    target_factor: Optional[int] = 4,
    rng: Optional[np.random.Generator] = None,
    label_rule: Optional[str] = None,
    root=None,
    normalize: bool = True,
):
    """
    Returns (train, test) for `kind` (see data.download.list_datasets()).

    n_samples, target_factor, rng and label_rule only apply to shapes3d, which
    has no canonical split; the other datasets ship their own. label_rule names
    an entry of LABEL_RULES that relabels the target factor (e.g. "parity").
    root (cache directory) and normalize (False: images in [0, 1]) are shapes3d-only too.
    """
    kwargs = {}
    if kind == "shapes3d":
        kwargs = dict(n_samples=n_samples, target_factor=target_factor, rng=rng,
                      root=root, normalize=normalize)
        kwargs["label_map"] = _label_map(target_factor, label_rule)
    elif label_rule is not None:
        raise ValueError("label_rule is only supported for shapes3d")
    out = get_dataset(kind, **kwargs)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise TypeError(f"get_dataset('{kind}') did not return a (train, test) split.")
    return out


def relabel_dataset(dataset, target_factor, label_rule=None):
    """
    `dataset` labelled by Shapes3D factor `target_factor` (relabelled with
    `label_rule` if given), sharing its images. A no-op when the dataset
    already uses that factor, so it is safe to call on every task.
    """
    if getattr(dataset, "target_factor", None) == target_factor:
        return dataset
    if not hasattr(dataset, "relabeled"):
        raise ValueError(f"{type(dataset).__name__} cannot be relabelled by a latent factor")
    return dataset.relabeled(target_factor, label_map=_label_map(target_factor, label_rule))


def _label_map(target_factor, label_rule):
    if label_rule is None:
        return None
    if target_factor is None:
        raise ValueError("label_rule needs a target_factor to relabel")
    if label_rule not in LABEL_RULES:
        raise ValueError(f"Unknown label_rule '{label_rule}'. Available: {list(LABEL_RULES)}")
    return LABEL_RULES[label_rule](SHAPES3D_FACTOR_N_VALUES[target_factor])


def dataset_spec(dataset):
    """(C, H, W) of one image and the number of classes, inferred from the data."""
    x, _ = dataset[0]
    if x.dim() == 3:
        c, h, w = x.shape
    elif x.dim() == 2:
        c, (h, w) = 1, x.shape
    else:
        raise ValueError(f"Expected image-shaped samples, got shape {tuple(x.shape)}")

    n_classes = getattr(dataset, "n_classes", None)
    if n_classes is None and getattr(dataset, "classes", None) is not None:
        n_classes = len(dataset.classes)
    if n_classes is None:
        labels = [int(dataset[i][1]) for i in range(min(len(dataset), 10_000))]
        n_classes = max(labels) + 1
    return (c, h, w), int(n_classes)


def build_loaders(train_set, test_set, batch_size, num_workers=0):
    """
    (train_loader, test_loader): shuffled training batches of `batch_size`, and
    unshuffled test batches of at least 256.

    Python 3.14 starts worker processes with forkserver, which PICKLES the
    dataset into every worker -- for an in-memory Shapes3D split that is ~6 GB
    copied per worker and a certain OOM. fork shares the pages instead (the
    workers never touch CUDA, so forking after CUDA init is safe here).
    """
    import torch

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, **_loader_kwargs(num_workers))
    return train_loader, build_eval_loader(test_set, batch_size, num_workers)


def build_eval_loader(dataset, batch_size, num_workers=0):
    """Unshuffled batches of at least 256, for measuring loss/accuracy on any split."""
    import torch

    return torch.utils.data.DataLoader(
        dataset, batch_size=max(batch_size, 256), shuffle=False, **_loader_kwargs(num_workers))


def build_tensor_loader(*tensors, batch_size, shuffle=False, generator=None):
    """
    Batches of in-memory tensors (e.g. cached representations and their labels),
    indexed a whole batch at a time: per-item collation would dominate the cost of
    training a linear layer. Tensors stay on their device.
    """
    import torch
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset

    dataset = TensorDataset(*tensors)
    sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=BatchSampler(sampler, batch_size, drop_last=False),
                      batch_size=None)


def _loader_kwargs(num_workers):
    mp_context = "fork" if num_workers > 0 else None
    return dict(num_workers=num_workers, multiprocessing_context=mp_context,
                persistent_workers=num_workers > 0)
