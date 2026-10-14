"""
utils/stream_probe.py — Re-measuring the checkpoints of a task stream
(experiments/run_tasks_stream.py) on the data each task was trained on.

A stream run directory holds args.json, checkpoints/init.pt and one task_<NNN>/ per
task with its pixel permutation and its end-of-epoch checkpoints. The helpers here
rebuild the run's training split exactly as the stream built it and feed it to a
checkpoint the way the task presented it (normalised, pixels permuted).

Used by the GPU measuring steps of visualization_scripts/stream_*.py and by
experiments/replay_stream_unit_activity.py.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch

from data.download import SHAPES3D_MEAN, SHAPES3D_STD
from data.factory import build_dataset
from utils.checkpoints import load_model_from_checkpoint


def find_run(stream_dir, seed) -> Path:
    """The run directory of `seed` inside a stream results directory."""
    for r in glob.glob(str(Path(stream_dir) / "stream_*_seed*")):
        if r.rsplit("seed", 1)[1] == str(seed):
            return Path(r)
    raise SystemExit(f"no stream_*_seed{seed} in {stream_dir}")


def task_dirs(run) -> list[Path]:
    """The task_<NNN> directories of a run, in task order."""
    return [Path(t) for t in sorted(glob.glob(f"{run}/task_*"))]


def task_checkpoints(task_dir) -> list[str]:
    """The end-of-epoch checkpoints of a task, in epoch order (the last is the end of the task)."""
    return sorted(glob.glob(f"{task_dir}/checkpoints/epoch_*.pt"))


def task_perm(task_dir) -> torch.Tensor:
    """The task's pixel permutation as a LongTensor."""
    return torch.from_numpy(np.load(f"{task_dir}/task_pixel_permutation.npy")).long()


def load_init_model(run, device):
    """The run's network at initialisation, on `device`."""
    model, _ = load_model_from_checkpoint(Path(run) / "checkpoints" / "init.pt")
    return model.to(device)


def task_training_images(run_dir, device):
    """The run's training split, rebuilt exactly as the stream built it, as uint8 NHWC on `device`."""
    a = json.load(open(Path(run_dir) / "args.json"))
    train, _ = build_dataset(a["dataset_kind"], n_samples=a["n_samples"], target_factor=a["target_factor"],
                             rng=np.random.default_rng(a["seed"]), label_rule=a["label_rule"])
    return train, torch.from_numpy(np.ascontiguousarray(train.images)).to(device)


def draw_probe(images, seed, n_probe, salt):
    """
    (idx, probe): `n_probe` images drawn once per seed, in index order. `salt` keeps the
    draws of different measurements independent; reuse a measurement's salt to get its probe.
    """
    n_probe = min(n_probe, len(images))
    idx = np.sort(np.random.default_rng([int(seed), salt]).choice(len(images), n_probe, replace=False))
    return idx, images[torch.from_numpy(idx).to(images.device)]


def task_batches(images_u8, perm, batch_size=8192):
    """The images as the network saw them in the task: CHW, normalised, pixels permuted by `perm`."""
    dev = images_u8.device
    mean = torch.tensor(SHAPES3D_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(SHAPES3D_STD, device=dev).view(1, 3, 1, 1)
    perm = perm.to(dev)
    for i in range(0, len(images_u8), batch_size):
        x = (images_u8[i:i + batch_size].permute(0, 3, 1, 2).float().div_(255.0) - mean) / std
        yield x.reshape(len(x), 3, -1)[:, :, perm].reshape(x.shape)
