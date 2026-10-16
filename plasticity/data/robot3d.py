"""
data/robot3d.py — MPI3D-realistic and Isaac3D, the robot-arm counterparts of Shapes3D: images
rendered from a full grid of discrete factors, every image's factor values known.

    mpi3d_realistic  1 036 800 images, 64x64 (the 512x512 renders downsampled by the authors)
                     object_color 6, object_shape 6, object_size 2, camera_height 3,
                     background_color 3, horizontal_axis 40, vertical_axis 40
    isaac3d          737 280 images, 128x128 (the downsampled release of the 512x512 renders)
                     object_shape 3, robot_x 8, robot_y 5, camera_height 4, object_scale 4,
                     lighting_intensity 4, lighting_y_dir 6, object_color 4, wall_color 4

Both are stored in row-major factor order (the first factor varies slowest), so an image's
factor indices are np.unravel_index(its index, factor_n_values). For Isaac3D that order is
the one of the authors' dataset_demo.py (their README's index formula lists it wrongly).

Feature tasks: every factor is labelled by its parity (even index -> 1), as in the Shapes3D
feature streams, except the robot arm's position (MPI3D horizontal/vertical axis, Isaac3D
robot_x/robot_y), labelled by a threshold in the middle: index >= K / 2 (FeatureUpperHalfTask).
`factor_label_rules` holds the rule of each factor. Odd K are unbalanced: parity of 3 values
is 2/3 vs 1/3, and robot_y (5 values) splits 3 / 2 (60 / 40).

Prepare once, on a node that can download (GPU nodes cannot); this writes
<DATA_ROOT>/<name>/prepared_<res>px/{images.npy, indices.npy, meta.json}: a random subset of
--n_keep images (default 480 000, the size of Shapes3D; 0 = all), resized to --resolution
(default 64, the Shapes3D resolution) and stored as one uint8 array the loader memory-maps:

    uv run python -m data.robot3d prepare mpi3d_realistic                 # downloads the npz (~12 GB)
    uv run python -m data.robot3d prepare isaac3d --source /path/to/Isaac3D_down128
        (the unzipped 128x128 release, downloaded by hand from the Google Drive link of
         github.com/NVlabs/High-res-disentanglement-datasets: a folder of 737 280 PNGs)

Then:
    train, test = get_robot3d("isaac3d", n_samples=50_000, rng=np.random.default_rng(0))
    train.latent_idx              # (N, 9) factor indices
    train.factor_label_rules      # ['parity', 'upper_half', 'upper_half', 'parity', ...]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data.download import DATA_ROOT


@dataclass(frozen=True)
class Robot3DSpec:
    factor_names: tuple
    factor_n_values: tuple
    threshold_factors: tuple            # labelled by index >= K / 2 instead of parity
    native_resolution: int

    @property
    def n_images(self) -> int:
        return int(np.prod(self.factor_n_values))

    @property
    def factor_label_rules(self) -> list:
        return ["upper_half" if f in self.threshold_factors else "parity"
                for f in range(len(self.factor_names))]


ROBOT3D = {
    "mpi3d_realistic": Robot3DSpec(
        factor_names=("object_color", "object_shape", "object_size", "camera_height",
                      "background_color", "horizontal_axis", "vertical_axis"),
        factor_n_values=(6, 6, 2, 3, 3, 40, 40),
        threshold_factors=(5, 6),
        native_resolution=64),
    "isaac3d": Robot3DSpec(
        factor_names=("object_shape", "robot_x", "robot_y", "camera_height", "object_scale",
                      "lighting_intensity", "lighting_y_dir", "object_color", "wall_color"),
        factor_n_values=(3, 8, 5, 4, 4, 4, 6, 4, 4),
        threshold_factors=(1, 2),
        native_resolution=128),
}

MPI3D_REALISTIC_URL = "https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main/mpi3d_realistic.npz"
DEFAULT_N_KEEP = 480_000

# as Shapes3D: images from [0, 1] to [-1, 1]
ROBOT3D_MEAN = (0.5, 0.5, 0.5)
ROBOT3D_STD = (0.5, 0.5, 0.5)


def _spec(name) -> Robot3DSpec:
    if name not in ROBOT3D:
        raise ValueError(f"Unknown robot dataset '{name}'. Available: {list(ROBOT3D)}")
    return ROBOT3D[name]


def prepared_dir(name, resolution=64, root=None) -> Path:
    """Where `prepare` writes `name` at `resolution`, and the loader reads it."""
    _spec(name)
    return Path(root or DATA_ROOT / name) / f"prepared_{resolution}px"


# ── preparing (login node) ────────────────────────────────────────────────────

def _npz_array(path, key):
    """
    Array `key` of the .npz at `path`: memory-mapped when stored uncompressed (np.load
    would read all ~12 GB of MPI3D into memory), loaded otherwise.
    """
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
    if info.compress_type == zipfile.ZIP_STORED:
        with open(path, "rb") as f:
            f.seek(info.header_offset)
            local = f.read(30)
            if local[:4] != b"PK\x03\x04":
                raise ValueError(f"{path}: bad zip local header for {key}.npy")
            name_len, extra_len = struct.unpack("<HH", local[26:30])
            f.seek(info.header_offset + 30 + name_len + extra_len)
            version = np.lib.format.read_magic(f)
            read_header = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}[version]
            shape, fortran, dtype = read_header(f)
            offset = f.tell()
        return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset,
                         order="F" if fortran else "C")
    print(f"{path}: {key} is compressed, loading it whole into memory")
    return np.load(path)[key]


def _resize(img: np.ndarray, resolution: int) -> np.ndarray:
    """(H, W, 3) uint8 resized to (resolution, resolution, 3) by area averaging."""
    if img.shape[0] == resolution and img.shape[1] == resolution:
        return img
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((resolution, resolution), Image.BOX))


def _load_png(args):
    path, resolution = args
    from PIL import Image
    with Image.open(path) as im:
        return _resize(np.asarray(im.convert("RGB")), resolution)


def _select(n_images, n_keep, seed):
    """Sorted indices of the images kept: all of them, or a random n_keep."""
    if not n_keep or n_keep >= n_images:
        return np.arange(n_images)
    return np.sort(np.random.default_rng(seed).choice(n_images, size=n_keep, replace=False))


def _write(name, resolution, indices, fill, meta, root=None, block=8192):
    """
    Writes images.npy (len(indices), res, res, 3) uint8 and indices.npy, filling the images
    a block at a time with fill(block_indices) -> (n, res, res, 3); renamed into place last,
    so an interrupted run leaves no half-written dataset behind.
    """
    out = prepared_dir(name, resolution, root)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"images.{os.getpid()}.tmp.npy"
    images = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                       shape=(len(indices), resolution, resolution, 3))
    for start in range(0, len(indices), block):
        images[start:start + block] = fill(indices[start:start + block])
        print(f"\r  {min(start + block, len(indices))}/{len(indices)} images", end="", flush=True)
    print()
    images.flush()
    del images
    np.save(out / "indices.npy", indices.astype(np.int64))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out / "images.npy")
    return out


def prepare_mpi3d_realistic(source=None, resolution=64, n_keep=DEFAULT_N_KEEP, seed=0, root=None):
    """
    Writes the prepared MPI3D-realistic; `source` is mpi3d_realistic.npz, downloaded to
    <DATA_ROOT>/mpi3d_realistic/ when not given.
    """
    spec = _spec("mpi3d_realistic")
    if source is None:
        source = Path(root or DATA_ROOT / "mpi3d_realistic") / "mpi3d_realistic.npz"
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading MPI3D-realistic (~12 GB) -> {source}")
            part = source.with_suffix(".part")
            urllib.request.urlretrieve(MPI3D_REALISTIC_URL, filename=str(part))
            os.replace(part, source)
    raw = _npz_array(source, "images")
    raw = raw.reshape(-1, *raw.shape[-3:])                    # (N, 64, 64, 3), whatever the stored shape
    if len(raw) != spec.n_images:
        raise ValueError(f"{source}: {len(raw)} images, expected {spec.n_images}")
    indices = _select(spec.n_images, n_keep, seed)
    fill = lambda idx: np.stack([_resize(np.asarray(raw[i]), resolution) for i in idx]) \
        if resolution != raw.shape[1] else np.asarray(raw[idx])
    meta = dict(dataset="mpi3d_realistic", source=str(source), resolution=resolution,
                n_keep=len(indices), seed=seed)
    return _write("mpi3d_realistic", resolution, indices, fill, meta, root)


def prepare_isaac3d(source, resolution=64, n_keep=DEFAULT_N_KEEP, seed=0, root=None, workers=8):
    """
    Writes the prepared Isaac3D from `source`, the unzipped release: a folder holding the
    737 280 PNGs (directly or in images/), named by their zero-padded index.
    """
    from multiprocessing import Pool

    spec = _spec("isaac3d")
    source = Path(source)
    folder = source / "images" if (source / "images").is_dir() else source
    files = sorted(folder.glob("*.png"))
    if len(files) != spec.n_images:
        raise ValueError(f"{folder}: {len(files)} PNGs, expected {spec.n_images}")
    indices = _select(spec.n_images, n_keep, seed)
    with Pool(workers) as pool:
        fill = lambda idx: np.stack(pool.map(_load_png, [(files[i], resolution) for i in idx],
                                             chunksize=64))
        meta = dict(dataset="isaac3d", source=str(source), resolution=resolution,
                    n_keep=len(indices), seed=seed)
        return _write("isaac3d", resolution, indices, fill, meta, root)


# ── loading ───────────────────────────────────────────────────────────────────

class FactorImageDataset(torch.utils.data.Dataset):
    """
    In-memory split of a factor dataset yielding (image, target) pairs, as
    data.download.Shapes3DImageDataset: uint8 (N, H, W, 3) images, float CHW tensors on access.

    target_factor: int -> that factor's index (through label_map if given); None -> all of them.
    """

    def __init__(self, images, latent_idx, spec: Robot3DSpec, target_factor: Optional[int] = None,
                 normalize: bool = True, flatten: bool = False, label_map=None):
        n_factors = len(spec.factor_names)
        if target_factor is not None and not 0 <= target_factor < n_factors:
            raise ValueError(f"target_factor must be in [0, {n_factors}) or None, got {target_factor}")
        if label_map is not None:
            if target_factor is None:
                raise ValueError("label_map needs an int target_factor")
            if len(label_map) != spec.factor_n_values[target_factor]:
                raise ValueError(f"label_map has {len(label_map)} entries, factor {target_factor} "
                                 f"has {spec.factor_n_values[target_factor]} values")
            label_map = np.asarray(label_map, dtype=np.int64)
        self.images = images
        self.latent_idx = latent_idx
        self.spec = spec
        self.target_factor = target_factor
        self.label_map = label_map
        self.normalize = normalize
        self.flatten = flatten
        self.factor_names = list(spec.factor_names)
        self.factor_n_values = list(spec.factor_n_values)
        self.factor_label_rules = spec.factor_label_rules
        self._mean = torch.tensor(ROBOT3D_MEAN).view(3, 1, 1)
        self._std = torch.tensor(ROBOT3D_STD).view(3, 1, 1)

    def relabeled(self, target_factor, label_map=None) -> "FactorImageDataset":
        """The same images labelled by another factor; the arrays are shared, not copied."""
        return FactorImageDataset(self.images, self.latent_idx, self.spec, target_factor,
                                  self.normalize, self.flatten, label_map)

    def subset(self, idx) -> "FactorImageDataset":
        """The images at `idx` (copied), with the same labelling and preprocessing."""
        return FactorImageDataset(self.images[idx], self.latent_idx[idx], self.spec, self.target_factor,
                                  self.normalize, self.flatten, self.label_map)

    @property
    def n_classes(self) -> Optional[int]:
        if self.target_factor is None:
            return None
        if self.label_map is not None:
            return int(self.label_map.max()) + 1
        return self.factor_n_values[self.target_factor]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = torch.from_numpy(np.asarray(self.images[i])).permute(2, 0, 1).float().div_(255.0)
        if self.normalize:
            img = (img - self._mean) / self._std
        if self.flatten:
            img = img.reshape(-1)
        z = torch.from_numpy(np.asarray(self.latent_idx[i]))
        target = z if self.target_factor is None else int(z[self.target_factor])
        if self.label_map is not None:
            target = int(self.label_map[target])
        return img, target


def get_robot3d(name, root=None, resolution=64, train_frac=0.9, n_samples=None, target_factor=None,
                normalize=True, flatten=False, rng=None, label_map=None):
    """
    (train, test) of the prepared `name` (see the module docstring), as get_shapes3d: n_samples
    images drawn at random from the prepared ones (all by default), split by a permutation from
    `rng` (default np.random.default_rng(0)), the first train_frac of it for training.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    spec = _spec(name)
    folder = prepared_dir(name, resolution, root)
    if not (folder / "images.npy").exists():
        raise FileNotFoundError(
            f"{folder}/images.npy not found: prepare {name} first, on a node that can download:\n"
            f"  uv run python -m data.robot3d prepare {name}"
            + (" --source /path/to/Isaac3D_down128" if name == "isaac3d" else ""))
    rng = rng if rng is not None else np.random.default_rng(0)

    all_images = np.load(folder / "images.npy", mmap_mode="r")
    indices = np.load(folder / "indices.npy")
    n_total = len(all_images)
    if n_samples is not None and n_samples < n_total:
        sel = np.sort(rng.choice(n_total, size=n_samples, replace=False))
        images = np.empty((len(sel),) + all_images.shape[1:], dtype=np.uint8)
        for start in range(0, len(sel), 16384):               # sorted reads, one block in flight
            images[start:start + 16384] = all_images[sel[start:start + 16384]]
    else:
        sel = np.arange(n_total)
        images = np.array(all_images)
    latent_idx = np.stack(np.unravel_index(indices[sel], spec.factor_n_values), axis=1).astype(np.int64)

    perm = rng.permutation(len(images))
    n_train = int(round(len(images) * train_frac))

    def _split(idx):
        return FactorImageDataset(images[idx], latent_idx[idx], spec, target_factor,
                                  normalize, flatten, label_map)

    return _split(perm[:n_train]), _split(perm[n_train:])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare MPI3D-realistic or Isaac3D (see the module docstring)")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("dataset", choices=list(ROBOT3D))
    prep.add_argument("--source", default=None,
                      help="mpi3d_realistic: the .npz (default: download it); isaac3d: the unzipped PNG folder")
    prep.add_argument("--resolution", type=int, default=64)
    prep.add_argument("--n_keep", type=int, default=DEFAULT_N_KEEP, help="random images kept; 0 = all")
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--workers", type=int, default=8, help="isaac3d: processes decoding the PNGs")
    args = parser.parse_args(argv)
    if args.dataset == "mpi3d_realistic":
        out = prepare_mpi3d_realistic(args.source, args.resolution, args.n_keep, args.seed)
    else:
        if args.source is None:
            parser.error("isaac3d: pass --source, the unzipped folder of the 128x128 release")
        out = prepare_isaac3d(args.source, args.resolution, args.n_keep, args.seed, workers=args.workers)
    print(f"prepared {args.dataset} -> {out}")


if __name__ == "__main__":
    main()
