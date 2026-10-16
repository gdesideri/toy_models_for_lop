"""
data/input_shift.py — Shift every input image by a fixed image, and the input statistics
that say how hard one gradient step pushes a first-layer unit on all images at once.

    stats = input_stats(train)                       # shared_push, unit_spread, push_over_spread, ...
    shift = shift_for(train, center=1.0, offset=0.0) # -1 x the train mean image (None: no shift)
    train, test = ShiftedInputs(train, shift), ShiftedInputs(test, shift)

A step on a first-layer unit moves its input on another image x' by the error times
x . x'; averaged over pairs of images, x . x' is the energy of the mean image, ||m||^2
(`shared_push`). Divided by how much a Kaiming-initialised unit's input varies across
images (`unit_spread` = sqrt(2 x mean per-input variance)), it says how far one unit of
error moves every image together, in spreads (`push_over_spread`).

`center` = alpha subtracts alpha x the train mean image (alpha = 1: centred inputs, no
shared push); `offset` = c adds c to every input (raises the shared push to
||m||^2 + 2c sum(m) + c^2 D; `offset_for_ratio` solves for a target push_over_spread).

Print the statistics of a dataset and the offset that reaches a target ratio:
    uv run python -m data.input_shift --dataset cifar10 --target_ratio 1849
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _moments(dataset, batch_size=1024):
    """(per-input mean, per-input mean of squares, image shape, number of images), float64.
    A DataLoader pass draws from the global torch RNG, so it runs on a forked one."""
    with torch.random.fork_rng(devices=[]):
        n, total, squares, shape = 0, 0.0, 0.0, None
        for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            shape = tuple(x.shape[1:])
            x = x.reshape(len(x), -1).double()
            n += len(x)
            total = total + x.sum(0)
            squares = squares + x.pow(2).sum(0)
    return total / n, squares / n, shape, n


def mean_image(dataset) -> torch.Tensor:
    """The mean of the images of `dataset`, as one image (float32)."""
    m, _, shape, _ = _moments(dataset)
    return m.reshape(shape).float()


def input_stats(dataset) -> dict:
    """The shared push of `dataset`'s images and what it is measured against (plain floats)."""
    m, squares, _, n = _moments(dataset)
    shared_push = float(m.pow(2).sum())
    mean_energy = float(squares.sum())
    unit_spread = math.sqrt(2 * float((squares - m.pow(2)).clamp(min=0).mean()))
    return dict(shared_push=shared_push, mean_energy=mean_energy,
                shared_fraction=shared_push / mean_energy if mean_energy else 0.0,
                unit_spread=unit_spread,
                push_over_spread=shared_push / unit_spread if unit_spread else float("inf"),
                mean_total=float(m.sum()), n_inputs=m.numel(), n_images=n)


def shift_for(dataset, center: float, offset: float):
    """The image added to every input: -center x the mean image of `dataset` + offset.
    None when both are 0 (inputs left as they are)."""
    if not center and not offset:
        return None
    m = mean_image(dataset)
    return -center * m + offset


def offset_for_ratio(stats: dict, ratio: float) -> float:
    """The constant c whose addition to every input makes push_over_spread equal `ratio`
    (the spread does not change): c^2 D + 2c sum(m) + ||m||^2 = ratio x spread."""
    d, s, m2 = stats["n_inputs"], stats["mean_total"], stats["shared_push"]
    target = ratio * stats["unit_spread"]
    disc = s * s - d * (m2 - target)
    if disc < 0:
        raise ValueError(f"no constant offset reaches push_over_spread {ratio}")
    return (-s + math.sqrt(disc)) / d


class ShiftedInputs(Dataset):
    """`dataset` with `shift` added to every input image; every other attribute
    (latent_idx, targets, factor_names, ...) is the wrapped dataset's."""

    def __init__(self, dataset, shift):
        self.dataset = dataset
        self.shift = torch.as_tensor(shift, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return x + self.shift.reshape(x.shape), y

    def __getattr__(self, name):
        # only called for attributes the wrapper itself lacks; never for its own fields or
        # protocol names, so copying and unpickling cannot recurse
        if name in ("dataset", "shift") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.dataset, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--n_samples", type=int, default=None, help="shapes3d only")
    parser.add_argument("--center", type=float, default=0.0)
    parser.add_argument("--target_ratio", type=float, nargs="*", default=[])
    args = parser.parse_args()

    from data.factory import build_dataset
    train, _ = build_dataset(args.dataset, n_samples=args.n_samples, rng=np.random.default_rng(0))
    shift = shift_for(train, args.center, 0.0)
    if shift is not None:
        train = ShiftedInputs(train, shift)
    stats = input_stats(train)
    print(f"{args.dataset} (center {args.center:g}): " +
          ", ".join(f"{k} {v:.4g}" for k, v in stats.items()))
    for r in args.target_ratio:
        print(f"  push_over_spread {r:g}: --input_offset {offset_for_ratio(stats, r):.4f}")


if __name__ == "__main__":
    main()
