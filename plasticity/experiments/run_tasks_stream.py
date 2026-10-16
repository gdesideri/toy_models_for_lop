"""
experiments/run_tasks_stream.py — One network trained on a stream of tasks, each
task starting from the weights the previous one ended with.

A task is a label and a pixel permutation of the dataset. Three families:
    permuted pixels : a fresh permutation per task (--perm_percentage, default 100%),
                      the label fixed by --target_factor and --label_rule
    feature switches: --perm_percentage 0 --task_factors 0 1 2 3 4 5 --label_rule parity,
                      task n labelled by the parity of Shapes3D factor task_factors[n]; also on
                      mpi3d_realistic / isaac3d (data/robot3d.py), where the robot arm's position
                      factors are labelled by a threshold in the middle (index >= K/2) instead
    PCA factors     : --perm_percentage 0 --pca_bins K [--pca_components n --pca_score s
                      --pca_label_rule r], on any dataset (mnist, cifar10, ...): every image is scored on
                      the first n principal components of the raw train pixels (s = projection, or cosine:
                      the cosine similarity of the centred image with the component) and each score is
                      cut into K equal-count bins (K even; data/pca_factors.py); task n labels the image's
                      bin on component task_factors[n] (default: every component 0..n-1, n = 6) by rule r:
                      parity (even bin -> 1) or upper_half (bin >= K/2 -> 1)
    cluster factors : --perm_percentage 0 --cluster_factors F [--clusters C --cluster_dims D], on any
                      dataset: F categorical factors, each a k-means clustering (C clusters) of the
                      train images' first D pixel-PCA coordinates, fitted so that each factor is
                      non-redundant with the earlier ones; task n is labelled by the parity of the
                      image's cluster id on factor task_factors[n] (default: every factor; ids are
                      renumbered so the parity classes are about balanced; data/cluster_factors.py)
    class attributes: --perm_percentage 0 --class_tasks animal four_legs fur wings wheels, on cifar10:
                      task n is a semantic binary attribute of the image's class (e.g. wings =
                      airplane or bird vs the rest), trained and tested on a class-balanced subset
                      of at most --per_label images per label (tasks/class_attributes.py)
Optimizer defaults to plain SGD, momentum 0.

Run with:
    uv run python -m experiments.run_tasks_stream --dataset_kind shapes3d --target_factor 5 \
        --label_rule parity --model_kind mlp_2x30 --n_tasks 10 --num_epochs 10 --batch_size 256
    uv run python -m experiments.run_tasks_stream --dataset_kind mnist --perm_percentage 0 --pca_bins 8 \
        --model_kind mlp_2x30 --n_tasks 60 --num_epochs 10 --batch_size 256

    uv run python -m experiments.run_tasks_stream --dataset_kind mnist --perm_percentage 0 --cluster_factors 6 \
        --model_kind mlp_2x30 --n_tasks 60 --num_epochs 10 --batch_size 256
    uv run python -m experiments.run_tasks_stream --dataset_kind cifar10 --perm_percentage 0 \
        --class_tasks animal four_legs fur wings wheels --model_kind mlp_2x30 --n_tasks 500 \
        --num_epochs 10 --batch_size 256

Writes results/<save_dir>/stream_<dataset>_<model>_perm<p>_<label>_seed<k>/
(PCA runs: stream_<dataset>_<model>_perm000_pca<n>_bins<K>[_cos][_upper_half]_seed<k>/;
cluster runs: stream_<dataset>_<model>_perm000_clu<F>x<C>_d<D>_seed<k>/;
class-attribute runs: stream_<dataset>_<model>_perm000_attrs-<a>-<b>-..._seed<k>/):
    args.json, summary.json        the settings; final metrics of every task so far
    checkpoints/init.pt            the network before the first task
    task_<n>/                      history.json (per-step loss, per-epoch train loss / train acc /
                                   test loss / test acc), metrics.csv, summary.json,
                                   task_pixel_permutation.npy
    task_<n>/checkpoints/epoch_<e>.pt   snapshots over the epochs (utils.checkpoints); the last
                                   epoch is always saved and holds the task's final weights

Inputs: --center_inputs a subtracts a x the train mean image from every input and --input_offset c
adds c (data/input_shift.py), after the labels are made, so a shifted run has the same tasks as an
unshifted one. summary.json records `input_stats` (the shared push of the inputs the network sees).
Unit health: unless --health_every 0, the hidden units are measured on the whole training set (as
the network sees it in that task) at init and after every k-th task: dead and rarely active units,
effective rank, weight norms (utils.evaluation.unit_health) -> `init_health` and each task's `health`.
Measuring runs on a forked RNG, so it does not change training or --resume.

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
from data.factory import build_dataset, build_eval_loader, build_loaders, dataset_spec
from data.download import SHAPES3D_FACTOR_NAMES
from data.cluster_factors import cluster_cache_path, with_cluster_factors
from data.input_shift import ShiftedInputs, input_stats, shift_for
from data.pca_factors import pca_cache_path, with_pca_factors
from tasks.class_attributes import ClassAttributeTask, attribute_names
from tasks.feature_streams import FeatureParityTask, FeatureUpperHalfTask
from tasks.permuted_pixels import sample_pixel_permutation, permute_dataset
from models.factory import build_model
from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.evaluation import unit_health
from utils.results import run_dir_for, save_results, save_summary
from experiments.run_basic_task_training import train


RESUME_KEYS = ("dataset_kind", "n_samples", "target_factor", "label_rule", "perm_percentage",
               "model_kind", "optimizer", "learning_rate", "batch_size", "num_epochs", "seed",
               "task_factors", "features_per_task", "keep_features", "pca_components", "pca_bins",
               "cluster_factors", "clusters", "cluster_dims", "class_tasks", "per_label",
               "center_inputs", "input_offset", "pca_score", "pca_label_rule")
# the value a run saved before an argument existed had, implicitly
RESUME_DEFAULTS = {"features_per_task": 1, "pca_components": 6, "clusters": 8, "cluster_dims": 50,
                   "per_label": 10_000, "center_inputs": 0.0, "input_offset": 0.0,
                   "pca_score": "projection", "pca_label_rule": "parity"}


def changed_args(saved, params):
    """The settings in which `params` differ from a saved run's args.json."""
    return [k for k in RESUME_KEYS if saved.get(k, RESUME_DEFAULTS.get(k)) != getattr(params, k)]


def resolve_task_factors(params):
    """With --pca_bins or --cluster_factors: the task factors, every one by default, checked to exist."""
    if params.cluster_factors and params.pca_bins:
        raise SystemExit("--cluster_factors and --pca_bins are different factor sources; pass only one")
    if (params.pca_score != "projection" or params.pca_label_rule != "parity") and not params.pca_bins:
        raise SystemExit("--pca_score / --pca_label_rule only apply to PCA factors: pass --pca_bins")
    if params.class_tasks:
        if params.cluster_factors or params.pca_bins or params.task_factors:
            raise SystemExit("--class_tasks replaces --task_factors / --pca_bins / --cluster_factors; "
                             "pass only one task source")
        try:
            names = attribute_names(params.dataset_kind)
        except ValueError as e:
            raise SystemExit(f"--class_tasks: {e}")
        unknown = [t for t in params.class_tasks if t not in names]
        if unknown:
            raise SystemExit(f"--class_tasks {unknown}: unknown for {params.dataset_kind}; available: {names}")
        return None
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
            pca_cache_path(params.dataset_kind, params.pca_components, params.pca_bins, params.pca_score)
        train_set, test_set = with_pca_factors(train_set, test_set, params.pca_components,
                                               params.pca_bins, cache_path=cache, score=params.pca_score)
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
    if params.class_tasks:
        n_classes = 2
    shift = shift_for(train_set, params.center_inputs, params.input_offset)
    if shift is not None:
        # after the factor wrappers: the labels (and their caches) are those of the unshifted run
        train_set, test_set = ShiftedInputs(train_set, shift), ShiftedInputs(test_set, shift)
    stats = input_stats(train_set)
    print(f"inputs: shared push {stats['shared_push']:.1f}, unit spread {stats['unit_spread']:.2f}, "
          f"push/spread {stats['push_over_spread']:.1f}")
    probe_idx = np.sort(np.random.default_rng([params.seed, 11]).choice(
        len(train_set), min(5000, len(train_set)), replace=False))

    def measure_health(model, perm):
        """Unit health on the whole training set as the network sees it in a task (forked RNG)."""
        data = permute_dataset(train_set, perm)
        device = next(model.parameters()).device
        with torch.random.fork_rng(devices=[]):        # a DataLoader pass draws from the torch RNG
            batches = (x.to(device) for x, _ in build_eval_loader(data, 2048, params.num_workers))
            probe = torch.stack([data[int(i)][0] for i in probe_idx]).to(device)
            return unit_health(model, batches, probe)

    factor_names = attribute_names(params.dataset_kind) if params.class_tasks else \
        getattr(train_set, "factor_names", SHAPES3D_FACTOR_NAMES)

    if params.features_per_task != 1:
        raise SystemExit("--features_per_task: only 1 feature per task is supported")

    def factor_rule(factor):
        """How a factor labels a feature task: --pca_label_rule on PCA factors, else the dataset's
        own rule for that factor (data/robot3d.py: upper_half for the robot arm), else parity."""
        if params.pca_bins:
            return params.pca_label_rule
        rules = getattr(train_set, "factor_label_rules", None)
        return rules[factor] if rules else "parity"

    def feature_task(n):
        """Task n's labelling task in the feature family; None = the dataset's own label."""
        if params.class_tasks:
            return ClassAttributeTask(params.dataset_kind, params.class_tasks[n % len(params.class_tasks)],
                                      params.per_label, params.seed)
        if not params.task_factors:
            return None
        factor = params.task_factors[n % len(params.task_factors)]
        if factor_rule(factor) == "upper_half":
            return FeatureUpperHalfTask(factor)
        return FeatureParityTask([factor])

    if params.task_factors and not (params.pca_bins or params.cluster_factors) and (params.label_rule != "parity" or n_classes != 2):
        raise SystemExit("--task_factors: the feature tasks are parity tasks (2 classes, the head "
                         "carries over); pass --label_rule parity")
    run_dir = run_dir_for(params, prefix="stream_")
    stream, saved_summary = [], {}
    if params.resume and (run_dir / "summary.json").exists():
        saved = json.load(open(run_dir / "args.json"))
        changed = changed_args(saved, params)
        if changed:
            raise SystemExit(f"--resume: {run_dir} was run with different {changed}")
        saved_summary = json.load(open(run_dir / "summary.json"))
        stream = saved_summary["tasks"]
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
    model = model.to(params.device)
    extra = {"n_train": len(train_set), "n_test": len(test_set), "image_shape": [c, h, w],
             "n_classes": n_classes, "n_params": sum(p.numel() for p in model.parameters()),
             "input_stats": stats}
    if params.health_every:
        extra["init_health"] = saved_summary.get("init_health") if start else \
            measure_health(model, np.arange(h * w))
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

        description = task.name if params.class_tasks else \
            "+".join(factor_names[f] for f in factors) + \
            (" upper half" if task and factor_rule(factors[0]) == "upper_half" else " parity")
        print(f"── task {n + 1}/{params.n_tasks}  (label: {description}, "
              f"{len(base_train)}/{len(base_test)} images)" if task else
              f"── task {n + 1}/{params.n_tasks}")
        history = train(model, *loaders, params, on_epoch_end=snapshot)
        # model.pt would duplicate the last-epoch snapshot
        labels = {"factors": factors, **({"target_factor": factors[0]} if len(factors) == 1 else {})}
        if params.pca_bins:
            labels["explained_variance"] = [float(train_set.explained_variance[f]) for f in factors]
        if params.class_tasks:
            labels.update(attribute=task.name, n_task_train=len(base_train), n_task_test=len(base_test))
        if params.health_every and (n + 1) % params.health_every == 0:
            labels["health"] = measure_health(model, perm)
        if params.pca_bins or params.cluster_factors or params.class_tasks or \
                (task and hasattr(train_set, "factor_label_rules")):
            labels["class_balance"] = float(np.mean(base_train.labels))
        if task and hasattr(train_set, "factor_label_rules"):
            labels["rule"] = factor_rule(factors[0])
        save_results(task_dir, params, history, model, perm, {**extra, "task": n, **labels},
                     save_model=False)
        stream.append({"task": n, **labels,
                       "final_train_loss": history["train_loss"][-1],
                       "final_train_acc": history["train_acc"][-1],
                       "final_test_acc": history["test_acc"][-1],
                       "best_test_acc": max(history["test_acc"]),
                       **({"final_test_precision": history["test_precision"][-1]}
                          if history.get("test_precision") else {})})
        save_summary(run_dir, {**extra, "tasks": stream})     # the commit point of a task
        task_seconds.append(time.time() - t_task)

        # when finished training use the current model as the start for the next task
        # (`model` simply carries over. train() builds a new optimizer per task, which for
        # plain SGD with momentum 0 -- it keeps no state -- is the same as keeping one.)

    print(f"{len(stream)}/{params.n_tasks} tasks done; last final test accs "
          f"{[round(t['final_test_acc'], 4) for t in stream[-5:]]} -> {run_dir.resolve()}")
