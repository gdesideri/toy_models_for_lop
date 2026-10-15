"""The PCA options of experiments/run_tasks_stream.py: arguments, task factors, resume, run names."""
import pytest

from experiments.run_tasks_stream import changed_args, resolve_task_factors
from utils.args import parse_tasks_stream_args
from utils.results import run_dir_for


def test_pca_arguments_parse():
    p = parse_tasks_stream_args(["--dataset_kind", "mnist", "--pca_bins", "8"])
    assert p.pca_bins == 8 and p.pca_components == 6
    assert parse_tasks_stream_args([]).pca_bins is None


def test_pca_task_factors_default_to_every_component():
    p = parse_tasks_stream_args(["--pca_bins", "8", "--pca_components", "4"])
    assert resolve_task_factors(p) == [0, 1, 2, 3]


def test_explicit_task_factors_kept():
    p = parse_tasks_stream_args(["--pca_bins", "8", "--task_factors", "2", "0"])
    assert resolve_task_factors(p) == [2, 0]


def test_task_factor_beyond_the_components_rejected():
    p = parse_tasks_stream_args(["--pca_bins", "8", "--pca_components", "4", "--task_factors", "0", "4"])
    with pytest.raises(SystemExit, match="components 0-3"):
        resolve_task_factors(p)


def test_without_pca_task_factors_untouched():
    assert resolve_task_factors(parse_tasks_stream_args(["--task_factors", "0", "1"])) == [0, 1]
    assert resolve_task_factors(parse_tasks_stream_args([])) is None


def test_old_shapes3d_run_still_resumes():
    p = parse_tasks_stream_args(["--dataset_kind", "shapes3d", "--perm_percentage", "0", "--label_rule",
                                 "parity", "--task_factors", "0", "1", "2", "3", "4", "5"])
    saved = {k: v for k, v in vars(p).items() if k not in ("pca_components", "pca_bins", "features_per_task")}
    assert changed_args(saved, p) == []


def test_resume_with_other_bins_refused():
    saved = vars(parse_tasks_stream_args(["--pca_bins", "8"]))
    assert changed_args(saved, parse_tasks_stream_args(["--pca_bins", "4"])) == ["pca_bins"]


def _params(tmp_path, **settings):
    p = parse_tasks_stream_args(["--save_dir", str(tmp_path)])
    for k, v in settings.items():
        setattr(p, k, v)
    return p


def test_pca_run_name(tmp_path):
    p = _params(tmp_path, dataset_kind="mnist", model_kind="mlp_2x30", perm_percentage=0.0,
                pca_bins=8, pca_components=6, task_factors=[0, 1, 2, 3, 4, 5], seed=1)
    assert run_dir_for(p, prefix="stream_").name == "stream_mnist_mlp_2x30_perm000_pca6_bins8_seed1"


def test_pca_run_name_with_a_subset_of_components(tmp_path):
    p = _params(tmp_path, dataset_kind="mnist", model_kind="mlp_2x30", perm_percentage=0.0,
                pca_bins=8, pca_components=6, task_factors=[0, 2], seed=1)
    assert run_dir_for(p, prefix="stream_").name == "stream_mnist_mlp_2x30_perm000_pca6_bins8_features0-2_seed1"


def test_bin_counts_get_their_own_directories(tmp_path):
    names = {run_dir_for(_params(tmp_path, dataset_kind="mnist", pca_bins=k, task_factors=list(range(6))),
                         prefix="stream_").name for k in (2, 4, 8, 16)}
    assert len(names) == 4


def test_shapes3d_run_name_unchanged(tmp_path):
    p = _params(tmp_path, dataset_kind="shapes3d", model_kind="mlp_2x30", perm_percentage=0.0,
                task_factors=[0, 1, 2, 3, 4, 5], label_rule="parity", seed=0)
    assert run_dir_for(p, prefix="stream_").name == "stream_shapes3d_mlp_2x30_perm000_features012345_parity_seed0"


def test_cluster_arguments_parse():
    p = parse_tasks_stream_args(["--dataset_kind", "mnist", "--cluster_factors", "6"])
    assert (p.cluster_factors, p.clusters, p.cluster_dims) == (6, 8, 50)
    assert parse_tasks_stream_args([]).cluster_factors is None


def test_cluster_factors_with_pca_bins_refused():
    p = parse_tasks_stream_args(["--cluster_factors", "6", "--pca_bins", "8"])
    with pytest.raises(SystemExit, match="cluster_factors"):
        resolve_task_factors(p)


def test_cluster_task_factors_default_to_every_factor():
    p = parse_tasks_stream_args(["--cluster_factors", "4"])
    assert resolve_task_factors(p) == [0, 1, 2, 3]


def test_cluster_task_factor_out_of_range_rejected():
    p = parse_tasks_stream_args(["--cluster_factors", "4", "--task_factors", "0", "4"])
    with pytest.raises(SystemExit, match="factors 0-3 exist"):
        resolve_task_factors(p)


def test_cluster_run_name(tmp_path):
    p = _params(tmp_path, dataset_kind="mnist", model_kind="mlp_2x30", perm_percentage=0.0,
                cluster_factors=6, task_factors=list(range(6)), seed=0)
    assert run_dir_for(p, prefix="stream_").name == "stream_mnist_mlp_2x30_perm000_clu6x8_d50_seed0"


def test_cluster_run_name_with_a_subset_of_factors(tmp_path):
    p = _params(tmp_path, dataset_kind="mnist", model_kind="mlp_2x30", perm_percentage=0.0,
                cluster_factors=6, task_factors=[0, 2], seed=0)
    assert run_dir_for(p, prefix="stream_").name == "stream_mnist_mlp_2x30_perm000_clu6x8_d50_features0-2_seed0"


def test_old_pca_run_still_resumes():
    p = parse_tasks_stream_args(["--pca_bins", "8"])
    saved = {k: v for k, v in vars(p).items() if k not in ("cluster_factors", "clusters", "cluster_dims")}
    assert changed_args(saved, p) == []


def test_old_shapes3d_run_resumes_without_cluster_keys():
    p = parse_tasks_stream_args(["--dataset_kind", "shapes3d", "--perm_percentage", "0", "--label_rule",
                                 "parity", "--task_factors", "0", "1"])
    saved = {k: v for k, v in vars(p).items() if k not in ("cluster_factors", "clusters", "cluster_dims")}
    assert changed_args(saved, p) == []


def test_resume_with_other_clusters_refused():
    saved = vars(parse_tasks_stream_args(["--cluster_factors", "6"]))
    assert changed_args(saved, parse_tasks_stream_args(["--cluster_factors", "6", "--clusters", "4"])) == ["clusters"]
