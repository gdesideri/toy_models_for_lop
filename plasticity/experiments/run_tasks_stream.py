"""
experiments/run_tasks_stream.py — One network trained on a stream of tasks, each
task starting from the weights the previous one ended with.

A task is a label and a pixel permutation of the dataset. Three families:
    permuted pixels : a fresh permutation per task (--perm_percentage, default 100%),
                      the label fixed by --target_factor and --label_rule
    feature switches: --perm_percentage 0 --task_factors 0 1 2 3 4 5 --label_rule parity,
                      task n labelled by the parity of Shapes3D factor task_factors[n]
    PCA factors     : --perm_percentage 0 --pca_bins K [--pca_components n], on any dataset (mnist,
                      cifar10, ...): task n labelled by the parity of the image's bin on principal
                      component task_factors[n] (default: every component 0..n-1, n = 6), each
                      component of the raw train pixels cut into K equal-count bins (K even;
                      data/pca_factors.py)
    cluster factors : --perm_percentage 0 --cluster_factors F [--clusters C --cluster_dims D], on any
                      dataset: F categorical factors, each a k-means clustering (C clusters) of the
                      train images' first D pixel-PCA coordinates, fitted so that each factor is
                      non-redundant with the earlier ones; task n is labelled by the parity of the
                      image's cluster id on factor task_factors[n] (default: every factor; ids are
                      renumbered so the parity classes are about balanced; data/cluster_factors.py)
Optimizer defaults to plain SGD, momentum 0.

Run with:
    uv run python -m experiments.run_tasks_stream --dataset_kind shapes3d --target_factor 5 \
        --label_rule parity --model_kind mlp_2x30 --n_tasks 10 --num_epochs 10 --batch_size 256
    uv run python -m experiments.run_tasks_stream --dataset_kind mnist --perm_percentage 0 --pca_bins 8 \
        --model_kind mlp_2x30 --n_tasks 60 --num_epochs 10 --batch_size 256

    uv run python -m experiments.run_tasks_stream --dataset_kind mnist --perm_percentage 0 --cluster_factors 6 \
        --model_kind mlp_2x30 --n_tasks 60 --num_epochs 10 --batch_size 256

Writes results/<save_dir>/stream_<dataset>_<model>_perm<p>_<label>_seed<k>/
(PCA runs: stream_<dataset>_<model>_perm000_pca<n>_bins<K>_seed<k>/;
cluster runs: stream_<dataset>_<model>_perm000_clu<F>x<C>_d<D>_seed<k>/):
    args.json, summary.json        the settings; final metrics of every task so far
    checkpoints/init.pt            the network before the first task
    task_<n>/                      history.json (per-step loss, per-epoch train loss / train acc /
                                   test loss / test acc), metrics.csv, summary.json,
                                   task_pixel_permutation.npy
    task_<n>/checkpoints/epoch_<e>.pt   snapshots over the epochs (utils.checkpoints); the last
                                   epoch is always saved and holds the task's final weights

Long streams: --deadline_h stops cleanly between tasks before the wall-time limit, and
--resume continues a run directory after its last completed task. Every snapshot stores
the torch RNG state and the task draws are replayed, so a resumed stream continues
exactly as an uninterrupted one would.
"""
import json
import os
import sys
import time

T_START = time.time()

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_tasks_stream_args
from data.factory import build_dataset, build_loaders, dataset_spec
from data.download import SHAPES3D_FACTOR_NAMES
from data.cluster_factors import cluster_cache_path, with_cluster_factors
from data.pca_factors import pca_cache_path, with_pca_factors
from tasks.feature_streams import FeatureParityTask
from tasks.permuted_pixels import sample_pixel_permutation, permute_dataset
from models.factory import build_model
from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.results import run_dir_for, save_results, save_summary
from experiments.run_basic_task_training import train


RESUME_KEYS = ("dataset_kind", "n_samples", "target_factor", "label_rule", "perm_percentage",
               "model_kind", "optimizer", "learning_rate", "batch_size", "num_epochs", "seed",
               "task_factors", "features_per_task", "keep_features", "pca_components", "pca_bins",
               "cluster_factors", "clusters", "cluster_dims")
# the value a run saved before an argument existed had, implicitly
RESUME_DEFAULTS = {"features_per_task": 1, "pca_components": 6, "clusters": 8, "cluster_dims": 50}


def changed_args(saved, params):
    """The settings in which `params` differ from a saved run's args.json."""
    return [k for k in RESUME_KEYS if saved.get(k, RESUME_DEFAULTS.get(k)) != getattr(params, k)]


def resolve_task_factors(params):
    """With --pca_bins or --cluster_factors: the task factors, every one by default, checked to exist."""
    if params.cluster_factors and params.pca_bins:
        raise SystemExit("--cluster_factors and --pca_bins are different factor sources; pass only one")
    if not (params.pca_bins or params.cluster_factors):
        return params.task_factors
    n_factors = params.cluster_factors or params.pca_components
    factors = params.task_factors or list(range(n_factors))
    missing = [f for f in factors if not 0 <= f < n_factors]
    if missing and params.cluster_factors:
        raise SystemExit(f"--task_factors {missing}: only factors 0-{n_factors - 1} exist "
                         f"(--cluster_factors {n_factors})")
    if missing:
        raise SystemExit(f"--task_factors {missing}: only components 0-{n_factors - 1} exist "
                         f"(--pca_components {n_factors})")
    return factors


if __name__ == "__main__":
    params = parse_tasks_stream_args()
    params.task_factors = resolve_task_factors(params)
    rng = np.random.default_rng(params.seed)
    torch.manual_seed(params.seed)

    # load a dataset
    train_set, test_set = build_dataset(params.dataset_kind, n_samples=params.n_samples,
                                        target_factor=params.target_factor, rng=rng,
                                        label_rule=params.label_rule)
    (c, h, w), n_classes = dataset_spec(train_set)
    if params.pca_bins:
        # Shapes3D's split depends on the seed, so only fixed-split datasets are cached
        cache = None if params.dataset_kind == "shapes3d" else \
            pca_cache_path(params.dataset_kind, params.pca_components, params.pca_bins)
        train_set, test_set = with_pca_factors(train_set, test_set, params.pca_components,
                                               params.pca_bins, cache_path=cache)
        n_classes = 2
    if params.cluster_factors:
        cache = None if params.dataset_kind == "shapes3d" else \
            cluster_cache_path(params.dataset_kind, params.cluster_factors, params.clusters,
                               params.cluster_dims)
        train_set, test_set = with_cluster_factors(train_set, test_set, params.cluster_factors,
                                                   params.clusters, params.cluster_dims, cache_path=cache)
        n_classes = 2
        off_diagonal = train_set.factor_nmi[~np.eye(params.cluster_factors, dtype=bool)]
        print(f"cluster factors: class balance {np.round(train_set.class_balance, 3).tolist()}, "
              f"max off-diagonal NMI {off_diagonal.max() if off_diagonal.size else 0.0:.3f}")
    factor_names = getattr(train_set, "factor_names", SHAPES3D_FACTOR_NAMES)

    if params.features_per_task != 1:
        raise SystemExit("--features_per_task: only 1 feature per task is supported")

    def feature_task(n):
        """Task n's labelling task in the feature family; None = the dataset's own label."""
        if not params.task_factors:
            return None
        return FeatureParityTask([params.task_factors[n % len(params.task_factors)]])

    if params.task_factors and not (params.pca_bins or params.cluster_factors) and (params.label_rule != "parity" or n_classes != 2):
        raise SystemExit("--task_factors: the feature tasks are parity tasks (2 classes, the head "
                         "carries over); pass --label_rule parity")
    run_dir = run_dir_for(params, prefix="stream_")
    stream = []
    if params.resume and (run_dir / "summary.json").exists():
        saved = json.load(open(run_dir / "args.json"))
        changed = changed_args(saved, params)
        if changed:
            raise SystemExit(f"--resume: {run_dir} was run with different {changed}")
        stream = json.load(open(run_dir / "summary.json"))["tasks"]
    save_summary(run_dir, vars(params), name="args.json")

    # build a model
    model_kwargs = dict(arch=params.model_kind, d_in=c * h * w, in_channels=c, input_size=h,
                        n_classes=n_classes, activation=params.activation,
                        use_layer_norm=params.use_layer_norm)
    model = build_model(**model_kwargs)
    start = len(stream)
    if start == 0:
        save_checkpoint(run_dir / "checkpoints" / "init.pt", model, model_kwargs=model_kwargs,
                        task=None, epoch=0, torch_rng_state=torch.get_rng_state())
    else:
        # resume: the last completed task's final weights and RNG state, and replay the
        # task draws so that task `start` gets the permutation it would have had
        last = run_dir / f"task_{start - 1:03d}" / "checkpoints" / f"epoch_{params.num_epochs:03d}.pt"
        torch.set_rng_state(load_checkpoint(last, model)["meta"]["torch_rng_state"])
        for _ in range(start):
            sample_pixel_permutation(h * w, params.perm_percentage, rng)
        print(f"resuming after task {start - 1} ({last})")
    extra = {"n_train": len(train_set), "n_test": len(test_set), "image_shape": [c, h, w],
             "n_classes": n_classes, "n_params": sum(p.numel() for p in model.parameters())}
    print(f"{params.dataset_kind} {c}x{h}x{w} ({len(train_set)}/{len(test_set)}), {n_classes} classes | "
          f"{params.model_kind} | {params.n_tasks} tasks x {params.num_epochs} epochs, "
          f"{params.optimizer} lr {params.learning_rate} | {100 * params.perm_percentage:.0f}% permuted")

    task_seconds = []
    for n in range(start, params.n_tasks):
        if params.deadline_h is not None and task_seconds:
            if time.time() - T_START + 1.25 * max(task_seconds) > params.deadline_h * 3600:
                print(f"deadline of {params.deadline_h} h: stopping before task {n}; continue with --resume")
                break
        t_task = time.time()

        # sample a task from the dataset
        perm = sample_pixel_permutation(h * w, params.perm_percentage, rng)
        task = feature_task(n)
        factors = task.relevant_factors if task else [params.target_factor]
        base_train, base_test = (task.apply(train_set), task.apply(test_set)) if task else (train_set, test_set)
        task_train, task_test = permute_dataset(base_train, perm), permute_dataset(base_test, perm)
        loaders = build_loaders(task_train, task_test, params.batch_size, params.num_workers)
        task_dir = run_dir / f"task_{n:03d}"

        # train a model on the task, recording train loss / train acc / test acc over epochs,
        # save a snapshot of the network over the epochs
        def snapshot(epoch, model, n=n, task_dir=task_dir):
            if epoch % params.snapshot_every == 0 or epoch == params.num_epochs:
                save_checkpoint(task_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model,
                                model_kwargs=model_kwargs, task=n, epoch=epoch,
                                torch_rng_state=torch.get_rng_state())

        print(f"── task {n + 1}/{params.n_tasks}  (label: "
              f"{'+'.join(factor_names[f] for f in factors)} parity)" if task else
              f"── task {n + 1}/{params.n_tasks}")
        history = train(model, *loaders, params, on_epoch_end=snapshot)
        # model.pt would duplicate the last-epoch snapshot
        labels = {"factors": factors, **({"target_factor": factors[0]} if len(factors) == 1 else {})}
        if params.pca_bins:
            labels["explained_variance"] = [float(train_set.explained_variance[f]) for f in factors]
        if params.pca_bins or params.cluster_factors:
            labels["class_balance"] = float(np.mean(base_train.labels))
        save_results(task_dir, params, history, model, perm, {**extra, "task": n, **labels},
                     save_model=False)
        stream.append({"task": n, **labels,
                       "final_train_loss": history["train_loss"][-1],
                       "final_train_acc": history["train_acc"][-1],
                       "final_test_acc": history["test_acc"][-1],
                       "best_test_acc": max(history["test_acc"])})
        save_summary(run_dir, {**extra, "tasks": stream})     # the commit point of a task
        task_seconds.append(time.time() - t_task)

        # when finished training use the current model as the start for the next task
        # (`model` simply carries over. train() builds a new optimizer per task, which for
        # plain SGD with momentum 0 -- it keeps no state -- is the same as keeping one.)

    print(f"{len(stream)}/{params.n_tasks} tasks done; last final test accs "
          f"{[round(t['final_test_acc'], 4) for t in stream[-5:]]} -> {run_dir.resolve()}")
