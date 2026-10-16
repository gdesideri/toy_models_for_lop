"""A short MNIST stream on cosine PCA factors with upper-half labels, end to end through
experiments/run_tasks_stream.py; the PCA cache goes to a temporary data directory."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from data.download import DATA_ROOT
from tests.test_stream_shift_health import HAS_MNIST, PLASTICITY


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST not downloaded")
def test_mnist_cosine_upper_half_stream(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "mnist").symlink_to(DATA_ROOT / "mnist")
    subprocess.run([sys.executable, "-m", "experiments.run_tasks_stream", "--dataset_kind", "mnist",
                    "--perm_percentage", "0", "--pca_bins", "10", "--pca_score", "cosine",
                    "--pca_label_rule", "upper_half", "--model_kind", "mlp_2x30", "--num_epochs", "1",
                    "--batch_size", "256", "--n_tasks", "2", "--health_every", "0", "--seed", "0",
                    "--device", "cpu", "--save_dir", str(tmp_path / "out")],
                   cwd=PLASTICITY, check=True, capture_output=True, text=True,
                   env={**os.environ, "PLASTICITY_DATA_DIR": str(data_dir)})

    (run,) = (tmp_path / "out").glob("stream_*")
    assert run.name == "stream_mnist_mlp_2x30_perm000_pca6_bins10_cos_upper_half_seed0"
    tasks = json.load(open(run / "summary.json"))["tasks"]
    assert [t["factors"] for t in tasks] == [[0], [1]]
    # equal-count bins: the upper half of the train bins is exactly half of the train images
    np.testing.assert_allclose([t["class_balance"] for t in tasks], 0.5, atol=1e-3)
    assert (data_dir / "pca" / "mnist_pc6_bins10_cos.npz").exists()
