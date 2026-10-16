"""--center_inputs / --input_offset / --health_every in experiments/run_tasks_stream.py."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from data.download import DATA_ROOT
from experiments.run_tasks_stream import changed_args
from utils.args import parse_tasks_stream_args
from utils.results import run_dir_for

PLASTICITY = Path(__file__).resolve().parent.parent
HAS_MNIST = (DATA_ROOT / "mnist" / "MNIST" / "raw").exists()


def test_arguments_parse_with_defaults():
    p = parse_tasks_stream_args([])
    assert p.center_inputs == 0.0 and p.input_offset == 0.0 and p.health_every == 1
    p = parse_tasks_stream_args(["--center_inputs", "0.5", "--input_offset", "0.9113", "--health_every", "5"])
    assert p.center_inputs == 0.5 and p.input_offset == 0.9113 and p.health_every == 5


def _name(tmp_path, *flags):
    p = parse_tasks_stream_args(["--dataset_kind", "shapes3d", "--perm_percentage", "0", "--cluster_factors",
                                 "6", "--model_kind", "mlp_2x30", "--seed", "0", "--save_dir", str(tmp_path),
                                 *flags])
    return run_dir_for(p, prefix="stream_").name


def test_run_names_carry_the_shift(tmp_path):
    assert _name(tmp_path) == "stream_shapes3d_mlp_2x30_perm000_clu6x8_d50_seed0"
    assert _name(tmp_path, "--center_inputs", "1") == "stream_shapes3d_mlp_2x30_perm000_clu6x8_d50_c1_seed0"
    assert _name(tmp_path, "--center_inputs", "0.5") == "stream_shapes3d_mlp_2x30_perm000_clu6x8_d50_c0.5_seed0"
    assert _name(tmp_path, "--input_offset", "0.9113") == \
        "stream_shapes3d_mlp_2x30_perm000_clu6x8_d50_off0.9113_seed0"
    assert _name(tmp_path, "--health_every", "5") == "stream_shapes3d_mlp_2x30_perm000_clu6x8_d50_seed0"


def test_old_runs_still_resume_and_a_new_shift_is_refused():
    p = parse_tasks_stream_args(["--dataset_kind", "shapes3d", "--cluster_factors", "6"])
    saved = {k: v for k, v in vars(p).items() if k not in ("center_inputs", "input_offset", "health_every")}
    assert changed_args(saved, p) == []
    shifted = parse_tasks_stream_args(["--dataset_kind", "shapes3d", "--cluster_factors", "6", "--center_inputs", "1"])
    assert changed_args(saved, shifted) == ["center_inputs"]


def run_stream(save_dir, *flags):
    """A 2-task permuted-MNIST stream, 1 epoch per task; returns its run directory."""
    args = [sys.executable, "-m", "experiments.run_tasks_stream", "--dataset_kind", "mnist",
            "--model_kind", "mlp_2x30", "--num_epochs", "1", "--batch_size", "256", "--seed", "0",
            "--device", "cpu", "--save_dir", str(save_dir), *flags]
    if "--n_tasks" not in flags:
        args += ["--n_tasks", "2"]
    subprocess.run(args, cwd=PLASTICITY, check=True, capture_output=True, text=True)
    (run,) = Path(save_dir).glob("stream_*")
    return run


def final_weights(run, task):
    return torch.load(run / f"task_{task:03d}" / "checkpoints" / "epoch_001.pt", map_location="cpu")["model_state"]


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST not downloaded")
def test_measuring_health_does_not_change_training(tmp_path):
    with_health = run_stream(tmp_path / "a", "--health_every", "1")
    without = run_stream(tmp_path / "b", "--health_every", "0")
    a, b = final_weights(with_health, 1), final_weights(without, 1)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    summary = json.load(open(with_health / "summary.json"))
    assert "init_health" in summary and all("health" in t for t in summary["tasks"])
    first = summary["tasks"][0]["health"]
    assert [layer["units"] for layer in first["layers"]] == [30, 30] and first["n_images"] == 60000
    assert "effective_rank" in first
    assert "health" not in json.load(open(without / "summary.json"))["tasks"][0]


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST not downloaded")
def test_centred_run_has_no_shared_push_and_resumes(tmp_path):
    plain = json.load(open(run_stream(tmp_path / "plain", "--health_every", "0") / "summary.json"))
    assert plain["input_stats"]["push_over_spread"] == pytest.approx(192, rel=0.05)

    run = run_stream(tmp_path / "centred", "--center_inputs", "1", "--n_tasks", "1")
    assert run.name.endswith("_c1_seed0")
    first = json.load(open(run / "summary.json"))
    assert first["input_stats"]["shared_push"] < 1e-6
    run_stream(tmp_path / "centred", "--center_inputs", "1", "--n_tasks", "2", "--resume")
    resumed = json.load(open(run / "summary.json"))
    assert len(resumed["tasks"]) == 2
    assert resumed["init_health"] == first["init_health"]
