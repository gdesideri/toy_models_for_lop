"""
data/download.py — Lazy dataset download utilities.

None of these functions download anything on import. Call them explicitly
to fetch and cache datasets to DATA_ROOT.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import torch
from torchvision import datasets, transforms

DATA_ROOT = Path(os.environ.get("PLASTICITY_DATA_DIR", Path(__file__).parent / "_cache"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── MNIST ─────────────────────────────────────────────────────────────────────

def get_mnist(
    root: Optional[Path] = None,
    flatten: bool = False,
    normalize: bool = True,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Returns (train_dataset, test_dataset) for MNIST.
    Downloads to root/mnist/ on first call.
    """
    root = root or DATA_ROOT / "mnist"
    _ensure_dir(root)

    tfm_list: list = [transforms.ToTensor()]
    if normalize:
        tfm_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    if flatten:
        tfm_list.append(transforms.Lambda(lambda x: x.view(-1)))

    tfm = transforms.Compose(tfm_list)
    train = datasets.MNIST(root=str(root), train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)
    return train, test


# ── CIFAR-10 ──────────────────────────────────────────────────────────────────

def get_cifar10(
    root: Optional[Path] = None,
    normalize: bool = True,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Returns (train_dataset, test_dataset) for CIFAR-10.
    Downloads to root/cifar10/ on first call.
    """
    root = root or DATA_ROOT / "cifar10"
    _ensure_dir(root)

    tfm_list: list = [transforms.ToTensor()]
    if normalize:
        # per-channel means and stds computed on the training set
        tfm_list.append(
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            )
        )
    tfm = transforms.Compose(tfm_list)
    train = datasets.CIFAR10(root=str(root), train=True,  download=True, transform=tfm)
    test  = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tfm)
    return train, test


# ── Shapes3D ──────────────────────────────────────────────────────────────────

SHAPES3D_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
SHAPES3D_MD5 = "099a2078d58cec4daad0702c55d06868"


def get_shapes3d(root: Optional[Path] = None) -> Path:
    """
    Returns the path to the Shapes3D h5 file, downloading it if not present.

    The file is ~270 MB. Download may take a few minutes.

    Shapes3D latent factors (in order):
        0 - floor_hue       (10 values, [0, 1])
        1 - wall_hue        (10 values, [0, 1])
        2 - object_hue      (10 values, [0, 1])
        3 - object_size     (8 values,  [0.75, 1.25])
        4 - object_type     (4 values,  {cube, cylinder, sphere, capsule})
        5 - object_azimuth  (15 values, [0, 2π])

    Total images: 480 000  (64×64×3, uint8)
    """
    root = root or DATA_ROOT / "shapes3d"
    _ensure_dir(root)
    target = root / "3dshapes.h5"

    if target.exists():
        return target

    print(f"Downloading Shapes3D (~270 MB) → {target}")
    print("This may take a few minutes...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100.0, downloaded / total_size * 100)
        print(f"\r  {pct:.1f}%  ({downloaded // 1_000_000} / {total_size // 1_000_000} MB)",
              end="", flush=True)

    urllib.request.urlretrieve(SHAPES3D_URL, filename=str(target), reporthook=_progress)
    print()
    print(f"Saved to {target}")
    return target


def load_shapes3d_array(root: Optional[Path] = None):
    """
    Returns an open h5py file handle with keys:
        'images'  — (480000, 64, 64, 3)  uint8
        'labels'  — (480000, 6)           float32  (normalised latent values)

    Caller is responsible for closing the file.

    Example:
        f = load_shapes3d_array()
        images = f['images'][:]   # load all into memory
        labels = f['labels'][:]
        f.close()
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required to load Shapes3D. Install it with: pip install h5py"
        ) from e

    path = get_shapes3d(root)
    return h5py.File(str(path), "r")


# ── Registry ──────────────────────────────────────────────────────────────────

DATASETS = {
    "mnist":    get_mnist,
    "cifar10":  get_cifar10,
    "shapes3d": get_shapes3d,
}


def list_datasets() -> list[str]:
    return list(DATASETS.keys())


def get_dataset(name: str, **kwargs):
    """Generic dispatcher. Accepts names from list_datasets()."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    return DATASETS[name](**kwargs)
