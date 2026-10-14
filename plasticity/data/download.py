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

import numpy as np
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

SHAPES3D_FACTOR_NAMES = [
    "floor_hue", "wall_hue", "object_hue",
    "object_size", "object_type", "object_azimuth",
]
SHAPES3D_FACTOR_N_VALUES = [10, 10, 10, 8, 4, 15]
SHAPES3D_N_IMAGES = 480_000

# Shapes3D has no published per-channel statistics; 0.5/0.5 simply maps
# images from [0, 1] to [-1, 1]. Replace with measured values if you prefer.
SHAPES3D_MEAN = (0.5, 0.5, 0.5)
SHAPES3D_STD  = (0.5, 0.5, 0.5)


def download_shapes3d(root: Optional[Path] = None) -> Path:
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


def _shapes3d_latent_idx(latent_vals: "np.ndarray") -> "np.ndarray":
    """
    Convert the continuous latent values stored in the h5 file to integer indices:
    the sorted unique values of each factor map to 0, ..., K-1. Pass the labels of
    the WHOLE file, so that every factor shows all of its K values.
    """
    latent_idx = np.zeros_like(latent_vals, dtype=np.int64)
    for j, n in enumerate(SHAPES3D_FACTOR_N_VALUES):
        values, latent_idx[:, j] = np.unique(latent_vals[:, j], return_inverse=True)
        if len(values) != n:
            raise ValueError(f"Shapes3D factor {SHAPES3D_FACTOR_NAMES[j]} has {len(values)} "
                             f"distinct values, expected {n}")
    return latent_idx


class Shapes3DImageDataset(torch.utils.data.Dataset):
    """
    In-memory Shapes3D split yielding (image, target) pairs.

    Images are kept as uint8 (N, 64, 64, 3) and converted to float CHW tensors
    on access, so a split costs ~12 KB per image of RAM.

    target_factor:
        int   — the integer index of that latent factor (a class label).
        None  — a (6,) int64 tensor with the indices of all latent factors.

    label_map (only with an int target_factor):
        a sequence with one entry per value of that factor, mapping the factor
        index to the class label, e.g. [1, 0, 1, 0, ...] for a parity task.
    """

    def __init__(
        self,
        images,
        latent_idx,
        latent_vals=None,
        target_factor: Optional[int] = 4,
        normalize: bool = True,
        flatten: bool = False,
        label_map=None,
    ):
        if target_factor is not None and not 0 <= target_factor < len(SHAPES3D_FACTOR_NAMES):
            raise ValueError(
                f"target_factor must be in [0, {len(SHAPES3D_FACTOR_NAMES)}) or None, "
                f"got {target_factor}"
            )
        self.images = images
        self.latent_idx = latent_idx
        self.latent_vals = latent_vals
        self.target_factor = target_factor
        self.label_map = None
        if label_map is not None:
            if target_factor is None:
                raise ValueError("label_map needs an int target_factor")
            n_values = SHAPES3D_FACTOR_N_VALUES[target_factor]
            if len(label_map) != n_values:
                raise ValueError(f"label_map has {len(label_map)} entries, "
                                 f"factor {target_factor} has {n_values} values")
            self.label_map = np.asarray(label_map, dtype=np.int64)
        self.normalize = normalize
        self.flatten = flatten

        self._mean = torch.tensor(SHAPES3D_MEAN).view(3, 1, 1)
        self._std  = torch.tensor(SHAPES3D_STD).view(3, 1, 1)

    def relabeled(self, target_factor: Optional[int], label_map=None) -> "Shapes3DImageDataset":
        """The same images labelled by another factor; the arrays are shared, not copied."""
        return Shapes3DImageDataset(
            images=self.images, latent_idx=self.latent_idx, latent_vals=self.latent_vals,
            target_factor=target_factor, normalize=self.normalize, flatten=self.flatten,
            label_map=label_map,
        )

    def subset(self, idx) -> "Shapes3DImageDataset":
        """The images at `idx` (copied), with the same labelling and preprocessing."""
        return Shapes3DImageDataset(
            images=self.images[idx], latent_idx=self.latent_idx[idx],
            latent_vals=None if self.latent_vals is None else self.latent_vals[idx],
            target_factor=self.target_factor, normalize=self.normalize, flatten=self.flatten,
            label_map=self.label_map,
        )

    @property
    def n_classes(self) -> Optional[int]:
        if self.target_factor is None:
            return None
        if self.label_map is not None:
            return int(self.label_map.max()) + 1
        return SHAPES3D_FACTOR_N_VALUES[self.target_factor]

    @property
    def target_name(self) -> Optional[str]:
        if self.target_factor is None:
            return None
        return SHAPES3D_FACTOR_NAMES[self.target_factor]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = torch.from_numpy(np.asarray(self.images[i]))     # (64, 64, 3) uint8
        img = img.permute(2, 0, 1).float().div_(255.0)         # (3, 64, 64) in [0, 1]
        if self.normalize:
            img = (img - self._mean) / self._std
        if self.flatten:
            img = img.reshape(-1)

        z = torch.from_numpy(np.asarray(self.latent_idx[i]))
        target = z if self.target_factor is None else int(z[self.target_factor])
        if self.label_map is not None:
            target = int(self.label_map[target])
        return img, target


def _read_rows(dset, rows, block=16384):
    """
    dset[rows] for sorted row indices, read in contiguous blocks. h5py's own point
    selection over tens of thousands of scattered rows takes tens of GB of memory
    (it OOM-killed 60k-image runs at 43 GB); this peaks at one block plus the result.
    """
    out = np.empty((len(rows),) + dset.shape[1:], dtype=dset.dtype)
    j = 0
    for start in range(0, dset.shape[0], block):
        stop = min(start + block, dset.shape[0])
        lo, hi = np.searchsorted(rows, [start, stop])
        if hi > lo:
            out[j:j + hi - lo] = dset[start:stop][rows[lo:hi] - start]
            j += hi - lo
    return out


def get_shapes3d(
    root: Optional[Path] = None,
    train_frac: float = 0.9,
    n_samples: Optional[int] = None,
    target_factor: Optional[int] = 4,
    normalize: bool = True,
    flatten: bool = False,
    rng: Optional["np.random.Generator"] = None,
    label_map=None,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Returns (train_dataset, test_dataset) for Shapes3D.

    Downloads the h5 file to root/ on first call (~270 MB), then loads the
    images into memory and splits them randomly.

    Shapes3D has no canonical split, so one is drawn here: a permutation drawn
    from `rng`, with the first `train_frac` of it used for training.

    Args:
        root:          cache directory (default: DATA_ROOT/shapes3d).
        train_frac:    fraction of the loaded images used for training.
        n_samples:     load only this many images (drawn uniformly at random).
                       The full dataset is 480 000 images ≈ 5.9 GB of RAM as
                       uint8, so pass a smaller number unless you need it all.
        target_factor: latent factor used as the label (default 4 = object_type,
                       4 classes). None yields all six factor indices instead.
        label_map:     optional factor-index -> class map (see Shapes3DImageDataset).
        normalize:     rescale images from [0, 1] to [-1, 1] (SHAPES3D_MEAN/STD).
        flatten:       return images as flat (3*64*64,) vectors.
        rng:           np.random.Generator used for both subsampling and the
                       split (default: np.random.default_rng(0), so repeated
                       calls give the same split unless you pass your own).

    Use `download_shapes3d()` if you only want the path to the h5 file, or
    `load_shapes3d_array()` for the raw h5 handle.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if rng is None:
        rng = np.random.default_rng(0)

    f = load_shapes3d_array(root)
    try:
        n_total = f["images"].shape[0]

        # labels are small (~11 MB); read them all so the value→index mapping
        # always sees the full range of every factor, even when subsampling
        all_vals = np.asarray(f["labels"][:], dtype=np.float32)
        all_idx  = _shapes3d_latent_idx(all_vals)

        if n_samples is not None and n_samples < n_total:
            # h5py requires indices in increasing order
            sel = np.sort(rng.choice(n_total, size=n_samples, replace=False))
        else:
            sel = slice(None)

        images = f["images"][sel] if isinstance(sel, slice) else _read_rows(f["images"], sel)
    finally:
        f.close()

    latent_vals = all_vals[sel]
    latent_idx  = all_idx[sel]

    n = len(images)
    perm    = rng.permutation(n)
    n_train = int(round(n * train_frac))
    train_i, test_i = perm[:n_train], perm[n_train:]

    def _split(idx):
        return Shapes3DImageDataset(
            images=images[idx],
            latent_idx=latent_idx[idx],
            latent_vals=latent_vals[idx],
            target_factor=target_factor,
            normalize=normalize,
            flatten=flatten,
            label_map=label_map,
        )

    return _split(train_i), _split(test_i)


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

    path = download_shapes3d(root)
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
