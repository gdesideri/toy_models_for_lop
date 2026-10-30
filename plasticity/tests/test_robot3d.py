"""data/robot3d.py: preparing and loading MPI3D-realistic / Isaac3D, their label rules, and a short
feature stream on them through experiments/run_tasks_stream.py. The real files are never touched:
preparing runs on a tiny factor grid, the stream on a fake prepared dataset."""
import functools
import http.server
import io
import json
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import data.robot3d as robot3d
from data.factory import build_dataset
from data.robot3d import ROBOT3D, Robot3DSpec, get_robot3d
from tasks.feature_streams import FeatureParityTask, FeatureUpperHalfTask
from tests.test_stream_shift_health import PLASTICITY
from utils.results import run_dir_for

TINY = (2, 3, 2, 1, 1, 4, 4)                    # 192 images, MPI3D's factor layout


def test_specs():
    mpi, isaac = ROBOT3D["mpi3d_realistic"], ROBOT3D["isaac3d"]
    assert mpi.n_images == 1_036_800 and isaac.n_images == 737_280
    assert [mpi.factor_names[f] for f in mpi.threshold_factors] == ["horizontal_axis", "vertical_axis"]
    assert [isaac.factor_names[f] for f in isaac.threshold_factors] == ["robot_x", "robot_y"]
    assert isaac.factor_label_rules == ["parity", "upper_half", "upper_half"] + ["parity"] * 6


@pytest.mark.parametrize("save", [np.savez, np.savez_compressed])
def test_npz_array_memmaps_or_loads(tmp_path, save):
    images = np.arange(5 * 4 * 4 * 3, dtype=np.uint8).reshape(5, 4, 4, 3)
    save(tmp_path / "x.npz", images=images, other=np.zeros(3))
    out = robot3d._npz_array(tmp_path / "x.npz", "images")
    np.testing.assert_array_equal(out, images)
    assert isinstance(out, np.memmap) == (save is np.savez)


def _check_latents(train, test, spec):
    """Every image is filled with its own flat index: its factors must be that index unravelled."""
    for split in (train, test):
        index = split.images[:, 0, 0, 0].astype(np.int64)
        expected = np.stack(np.unravel_index(index, spec.factor_n_values), axis=1)
        np.testing.assert_array_equal(split.latent_idx, expected)


def test_prepare_and_load_mpi3d(tmp_path, monkeypatch):
    spec = Robot3DSpec(ROBOT3D["mpi3d_realistic"].factor_names, TINY, (5, 6), 8)
    monkeypatch.setitem(ROBOT3D, "mpi3d_realistic", spec)
    images = np.broadcast_to(np.arange(spec.n_images, dtype=np.uint8)[:, None, None, None],
                             (spec.n_images, 8, 8, 3)).reshape(*TINY, 8, 8, 3)   # stored factor-shaped
    np.savez(tmp_path / "mpi3d_realistic.npz", images=images)

    robot3d.prepare_mpi3d_realistic(tmp_path / "mpi3d_realistic.npz", resolution=8, n_keep=100, root=tmp_path)
    train, test = get_robot3d("mpi3d_realistic", root=tmp_path, resolution=8, n_samples=80,
                              normalize=False, rng=np.random.default_rng(1))
    assert (len(train), len(test)) == (72, 8)
    _check_latents(train, test, spec)
    x, y = train[0]
    assert x.shape == (3, 8, 8) and y is not None


ISAAC_TINY = (2, 2, 3, 1, 1, 1, 1, 2, 2)        # 48 images, Isaac3D's factor layout


def _png(i):
    buffer = io.BytesIO()
    Image.fromarray(np.full((16, 16, 3), i, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _isaac_zip(path, n):
    """The release's layout: Isaac3D/images/<index>.png, members in shuffled order, deflated."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Isaac3D/", b"")
        for i in np.random.default_rng(3).permutation(n):
            zf.writestr(f"Isaac3D/images/{i:06d}.png", _png(int(i)),
                        compress_type=zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED)
    return path


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files with HTTP range requests (bytes=a-b and bytes=a-), as Google Drive does."""

    def do_GET(self):
        data = Path(self.directory, self.path.lstrip("/")).read_bytes()
        start, end = self.headers["Range"].removeprefix("bytes=").split("-")
        start, end = int(start), int(end) if end else len(data) - 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start:end + 1])

    def log_message(self, *args):
        pass


@pytest.fixture
def isaac_tiny(monkeypatch):
    spec = Robot3DSpec(ROBOT3D["isaac3d"].factor_names, ISAAC_TINY, (1, 2), 16)
    monkeypatch.setitem(ROBOT3D, "isaac3d", spec)
    return spec


def test_prepare_isaac3d_from_pngs_downsampled(tmp_path, isaac_tiny):
    (tmp_path / "src" / "images").mkdir(parents=True)
    for i in range(isaac_tiny.n_images):
        (tmp_path / "src" / "images" / f"{i:06d}.png").write_bytes(_png(i))

    out = robot3d.prepare_isaac3d(tmp_path / "src", resolution=8, n_keep=0, root=tmp_path, workers=2)
    assert np.load(out / "images.npy").shape == (isaac_tiny.n_images, 8, 8, 3)
    _check_latents(*get_robot3d("isaac3d", root=tmp_path, resolution=8, normalize=False), isaac_tiny)


def test_prepare_isaac3d_from_a_local_zip(tmp_path, isaac_tiny):
    source = _isaac_zip(tmp_path / "Isaac3D.zip", isaac_tiny.n_images)
    robot3d.prepare_isaac3d(source, resolution=8, n_keep=30, root=tmp_path, workers=2, block=7)
    train, test = get_robot3d("isaac3d", root=tmp_path, resolution=8, normalize=False)
    assert len(train) + len(test) == 30
    _check_latents(train, test, isaac_tiny)


def test_prepare_isaac3d_streamed_by_range_requests(tmp_path, isaac_tiny):
    _isaac_zip(tmp_path / "Isaac3D.zip", isaac_tiny.n_images)
    handler = functools.partial(_RangeHandler, directory=str(tmp_path))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/Isaac3D.zip"
        robot3d.prepare_isaac3d(url, resolution=8, n_keep=0, root=tmp_path, workers=2, block=5)
        server.shutdown()
    _check_latents(*get_robot3d("isaac3d", root=tmp_path, resolution=8, normalize=False), isaac_tiny)


def test_interrupted_preparation_continues(tmp_path, isaac_tiny, monkeypatch):
    source = _isaac_zip(tmp_path / "Isaac3D.zip", isaac_tiny.n_images)
    calls = []
    member_bytes = robot3d._member_bytes

    def flaky(reader, info):
        calls.append(info.filename)
        if len(calls) == 20:
            raise ConnectionError("killed")
        return member_bytes(reader, info)

    monkeypatch.setattr(robot3d, "_member_bytes", flaky)
    with pytest.raises(ConnectionError):
        robot3d.prepare_isaac3d(source, resolution=8, n_keep=0, root=tmp_path, workers=2, block=8)
    folder = robot3d.prepared_dir("isaac3d", 8, tmp_path)
    assert not (folder / "images.npy").exists()
    assert json.loads((folder / "progress.json").read_text())["done"] == 16      # two whole blocks

    calls.clear()
    monkeypatch.setattr(robot3d, "_member_bytes", member_bytes)
    robot3d.prepare_isaac3d(source, resolution=8, n_keep=0, root=tmp_path, workers=2, block=8)
    assert not (folder / "progress.json").exists()
    _check_latents(*get_robot3d("isaac3d", root=tmp_path, resolution=8, normalize=False), isaac_tiny)


def test_missing_prepared_dataset_says_how_to_prepare(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.robot3d prepare isaac3d"):
        get_robot3d("isaac3d", root=tmp_path)


def _fake_prepared(root, name, n=600, res=8, seed=0):
    """A prepared folder at the 64px path holding n small random images at real factor indices."""
    rng = np.random.default_rng(seed)
    folder = robot3d.prepared_dir(name, 64, root / name)
    folder.mkdir(parents=True)
    np.save(folder / "images.npy", rng.integers(0, 256, (n, res, res, 3), dtype=np.uint8))
    np.save(folder / "indices.npy", np.sort(rng.choice(ROBOT3D[name].n_images, n, replace=False)))


def test_labels_parity_and_middle_threshold(tmp_path):
    _fake_prepared(tmp_path, "mpi3d_realistic")
    train, _ = build_dataset("mpi3d_realistic", target_factor=5, label_rule="parity",
                             root=tmp_path / "mpi3d_realistic")
    assert train.n_classes == 2
    axis = FeatureUpperHalfTask(5).apply(train)
    np.testing.assert_array_equal(axis.labels, (train.latent_idx[:, 5] >= 20).astype(int))
    color = FeatureParityTask([0]).apply(train)
    np.testing.assert_array_equal(color.labels, (train.latent_idx[:, 0] % 2 == 0).astype(int))


def test_run_name_lists_the_factors(tmp_path):
    params = SimpleNamespace(dataset_kind="isaac3d", model_kind="mlp_2x30", perm_percentage=0.0,
                             task_factors=list(range(9)), label_rule="parity", seed=2, run_name=None,
                             save_dir=str(tmp_path), target_factor=5)
    assert run_dir_for(params, prefix="stream_").name == \
        "stream_isaac3d_mlp_2x30_perm000_features012345678_parity_seed2"


def test_isaac3d_stream_uses_the_threshold_for_the_arm(tmp_path):
    data_dir = tmp_path / "data"
    _fake_prepared(data_dir, "isaac3d")
    subprocess.run([sys.executable, "-m", "experiments.run_tasks_stream", "--dataset_kind", "isaac3d",
                    "--perm_percentage", "0", "--task_factors", "0", "1", "2", "--label_rule", "parity",
                    "--target_factor", "5", "--model_kind", "mlp_2x30", "--num_epochs", "1",
                    "--batch_size", "64", "--n_tasks", "3", "--health_every", "0", "--seed", "0",
                    "--device", "cpu", "--save_dir", str(tmp_path / "out")],
                   cwd=PLASTICITY, check=True, capture_output=True, text=True,
                   env={**os.environ, "PLASTICITY_DATA_DIR": str(data_dir)})

    (run,) = (tmp_path / "out").glob("stream_*")
    assert run.name == "stream_isaac3d_mlp_2x30_perm000_features012_parity_seed0"
    tasks = json.load(open(run / "summary.json"))["tasks"]
    assert [(t["factors"], t["rule"]) for t in tasks] == \
        [([0], "parity"), ([1], "upper_half"), ([2], "upper_half")]
    # robot_y has 5 values: the threshold puts 2 of them (40%) in class 1
    assert abs(tasks[2]["class_balance"] - 0.4) < 0.08
