"""
experiments/calculate_repr_distance.py — How far is the representation learned on
one permuted-pixel task from what a second task needs?

    zero-shot     : loss of the task-1 model on task 2's training set, as is
    repr distance : acc(model trained on task 2) - acc(task-1 body + head retrained on task 2)

A distance of 0 means task 1's features are as good for task 2 as task 2's own;
the larger it is, the more of task 2 the task-1 representation cannot express.

A task is a label and a pixel permutation. One run = one pair of tasks, which
differ in their permutation (two independent ones drawn from the seed, at
--perm_percentage), in their label (--task2_target_factor), or both:
    permuted pixels : --perm_percentage 1.0
    feature pairs   : --perm_percentage 0 --target_factor 5 --task2_target_factor 4
                      --label_rule parity      (azimuth parity -> object-type parity)
    feature sets    : --perm_percentage 0 --label_rule parity --task1_factors 0 1 2 --task2_factors 1 2 3
                      [--set_rule parity|sum]: parity of several features, or "sum above median"
                      (tasks.feature_streams.FeatureParityTask / FeatureSumTask)
Sweep --seed for more pairs.

Run with:
    uv run python -m experiments.calculate_repr_distance --dataset_kind shapes3d \
        --target_factor 5 --model_kind mlp_2x50 --num_epochs 10 --batch_size 256
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_repr_distance_args
from data.factory import (build_dataset, build_loaders, build_eval_loader, dataset_spec,
                          relabel_dataset)
from tasks.permuted_pixels import sample_pixel_permutation, permute_dataset
from tasks.feature_streams import SET_RULES
from models.factory import build_model, reset_head, freeze_body
from utils.evaluation import evaluate
from utils.results import run_dir_for, save_results, save_summary
from visualization_scripts.task_training import plot_history, save_sample_images
from experiments.run_basic_task_training import train


if __name__ == "__main__":
    params = parse_repr_distance_args()
    rng = np.random.default_rng(params.seed)
    torch.manual_seed(params.seed)

    train_set, test_set = build_dataset(params.dataset_kind, n_samples=params.n_samples,
                                        target_factor=params.target_factor, rng=rng,
                                        label_rule=params.label_rule)
    (c, h, w), n_classes = dataset_spec(train_set)
    run_dir = run_dir_for(params, prefix="reprdist_")
    factors = {"task1": params.target_factor,
               "task2": (params.target_factor if params.task2_target_factor is None
                         else params.task2_target_factor)}
    feature_sets = None
    if params.task1_factors or params.task2_factors:
        if not (params.task1_factors and params.task2_factors) or n_classes != 2:
            raise SystemExit("--task1_factors and --task2_factors go together, on a 2-class dataset "
                             "(--label_rule parity); their labels come from --set_rule")
        rule = SET_RULES[params.set_rule]
        feature_sets = {"task1": rule(params.task1_factors), "task2": rule(params.task2_factors)}
    extra = {"n_train": len(train_set), "n_test": len(test_set),
             "image_shape": [c, h, w], "n_classes": n_classes,
             "task1_target_factor": factors["task1"], "task2_target_factor": factors["task2"],
             "label_rule": params.label_rule, "perm_percentage": params.perm_percentage}
    if feature_sets:
        extra.update(task1_factors=feature_sets["task1"].relevant_factors,
                     task2_factors=feature_sets["task2"].relevant_factors, set_rule=params.set_rule)

    def new_model():
        return build_model(params.model_kind, d_in=c * h * w, in_channels=c, input_size=h,
                           n_classes=n_classes, activation=params.activation,
                           use_layer_norm=params.use_layer_norm)

    # sample 2 pairs of tasks
    # (one pair per run: task 1 and task 2 -- each a label and a pixel permutation)
    tasks = {}
    for name in ("task1", "task2"):
        perm = sample_pixel_permutation(h * w, params.perm_percentage, rng)
        tr, te = train_set, test_set
        if feature_sets:
            tr, te = feature_sets[name].apply(train_set), feature_sets[name].apply(test_set)
        elif factors[name] != params.target_factor:
            tr = relabel_dataset(train_set, factors[name], params.label_rule)
            te = relabel_dataset(test_set, factors[name], params.label_rule)
        tasks[name] = dict(perm=perm, train=permute_dataset(tr, perm), test=permute_dataset(te, perm))
        save_sample_images(tasks[name]["train"], run_dir / f"sample_images_{name}.png", params,
                           n=params.n_sample_images, rng=np.random.default_rng(params.seed))
    if dataset_spec(tasks["task2"]["train"])[1] != n_classes:
        raise ValueError("task 1 and task 2 have different numbers of classes, so the "
                         "task-1 model cannot be evaluated zero-shot on task 2; use a "
                         "--label_rule (e.g. parity) that gives both the same classes")

    # train a model on task 1 and 2
    models, histories = {}, {}
    for name in ("task1", "task2"):
        print(f"── model trained on {name}")
        loaders = build_loaders(tasks[name]["train"], tasks[name]["test"],
                                params.batch_size, params.num_workers)
        models[name] = new_model()
        histories[name] = train(models[name], *loaders, params)
        save_results(run_dir / f"model_{name}", params, histories[name], models[name],
                     tasks[name]["perm"], extra)
        plot_history(histories[name], params, run_dir / f"model_{name}" / "curves.png")
    task2_loaders = loaders

    # calculate zero shot train loss of model trained on task 1 on task 2
    task2_train_eval = build_eval_loader(tasks["task2"]["train"], params.batch_size,
                                         params.num_workers)
    zero_shot_loss, zero_shot_acc = evaluate(models["task1"], task2_train_eval,
                                             nn.CrossEntropyLoss(), torch.device(params.device))
    print(f"── zero-shot, task-1 model on task 2 (train set): "
          f"loss {zero_shot_loss:.4f}  acc {zero_shot_acc:.4f}")

    # reset the head
    probe = freeze_body(reset_head(models["task1"], params.activation))

    # train head only on task 2
    print("── task-1 body, head retrained on task 2")
    head_params = argparse.Namespace(**{**vars(params),
                                        "num_epochs": params.head_epochs or params.num_epochs})
    histories["probe"] = train(probe, *task2_loaders, head_params)
    save_results(run_dir / "model_task1_body_task2_head", head_params, histories["probe"],
                 probe, tasks["task2"]["perm"], extra)
    plot_history(histories["probe"], head_params,
                 run_dir / "model_task1_body_task2_head" / "curves.png")

    # calculate accuracy of model trained on task 2 on task 2 - accuracy of model head retrained on task 2
    task2_acc = histories["task2"]["test_acc"][-1]
    probe_acc = histories["probe"]["test_acc"][-1]
    repr_distance = task2_acc - probe_acc

    # save both results
    save_summary(run_dir, {
        **extra,
        "zero_shot_task2_train_loss":     zero_shot_loss,
        "zero_shot_task2_train_acc":      zero_shot_acc,
        "repr_distance":                  repr_distance,
        "task2_model_test_acc":           task2_acc,
        "task1_body_task2_head_test_acc": probe_acc,
        "repr_distance_train_acc":        histories["task2"]["train_acc"][-1]
                                          - histories["probe"]["train_acc"][-1],
        "task1_model_test_acc":           histories["task1"]["test_acc"][-1],
        "head_epochs":                    head_params.num_epochs,
    })
    print(f"── zero-shot loss {zero_shot_loss:.4f} | repr distance {repr_distance:+.4f} "
          f"(task-2 model {task2_acc:.4f} - retrained head {probe_acc:.4f}) -> {run_dir.resolve()}")
