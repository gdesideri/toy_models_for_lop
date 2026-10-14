"""
utils/results.py — Where a run's outputs go, and writing them.

    run_dir = run_dir_for(params)      results/<save_dir>/<run name>/
    save_results(run_dir, params, history, model, task_perm, extra)
    save_csv(run_dir / "probes.csv", rows)    a list of flat dicts, one per row
    log = CsvLog(run_dir / "steps.csv")       the same, appended to as a run goes
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from data.download import SHAPES3D_FACTOR_NAMES


def run_dir_for(params, prefix: str = "") -> Path:
    """
    The output directory of a run, created if missing.

    The Shapes3D label is part of a run's identity: without it an
    object_azimuth sweep would silently overwrite the object_type one.
    """
    label = ""
    factors = getattr(params, "task_factors", None)
    t1, t2 = getattr(params, "task1_factors", None), getattr(params, "task2_factors", None)
    if params.dataset_kind == "shapes3d" and t1:
        label = f"_set{''.join(map(str, t1))}_to_set{''.join(map(str, t2))}_{getattr(params, 'set_rule', 'parity')}"
    elif params.dataset_kind == "shapes3d" and factors:
        label = "_features" + "".join(str(f) for f in factors)
        k = getattr(params, "features_per_task", 1) or 1
        keep = getattr(params, "keep_features", None)
        keep = keep if keep is not None else k - 1
        if k == 2:
            label += "_pairs"
        elif k == 3:
            label += f"_triples_keep{keep}"
        elif k > 3:
            label += f"_window{k}_keep{keep}"
        if getattr(params, "label_rule", None):
            label += f"_{params.label_rule}"
    elif params.dataset_kind == "shapes3d" and params.target_factor is not None:
        label = f"_{SHAPES3D_FACTOR_NAMES[params.target_factor]}"
        if getattr(params, "label_rule", None):
            label += f"_{params.label_rule}"
        task2 = getattr(params, "task2_target_factor", None)
        if task2 is not None and task2 != params.target_factor:
            label += f"_to_{SHAPES3D_FACTOR_NAMES[task2]}"
    run_name = params.run_name or prefix + (
        f"{params.dataset_kind}_{params.model_kind}_"
        f"perm{int(round(100 * params.perm_percentage)):03d}{label}_seed{params.seed}"
    )
    run_dir = Path(params.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_results(run_dir, params, history, model, task_perm, extra, save_model=True):
    """args.json, history.json, summary.json, metrics.csv, model.pt (unless save_model=False) and the task permutation."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(params), f, indent=2)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    with open(run_dir / "summary.json", "w") as f:
        json.dump({
            **extra,
            "final_train_loss": history["train_loss"][-1],
            "final_train_acc":  history["train_acc"][-1],
            "final_test_loss":  history["test_loss"][-1],
            "final_test_acc":   history["test_acc"][-1],
            "best_test_acc":    max(history["test_acc"]),
        }, f, indent=2)

    with open(run_dir / "metrics.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")
        for i, e in enumerate(history["epoch"]):
            f.write(f"{e},{history['train_loss'][i]},{history['train_acc'][i]},"
                    f"{history['test_loss'][i]},{history['test_acc'][i]}\n")

    if save_model:
        torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "task_pixel_permutation.npy", task_perm)


def save_summary(run_dir, summary: dict, name: str = "summary.json"):
    """Writes a flat dict of results to run_dir/name."""
    with open(Path(run_dir) / name, "w") as f:
        json.dump(summary, f, indent=2)


class CsvLog:
    """A CSV file written a batch of rows (flat dicts) at a time; the header comes from the first row."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = None

    def write(self, rows):
        rows = list(rows)
        if not rows:
            return
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(rows[0]))
            self._writer.writeheader()
        self._writer.writerows(rows)
        self._file.flush()

    def close(self):
        self._file.close()


def save_csv(path, rows):
    """Writes a list of flat dicts to `path` as CSV."""
    log = CsvLog(path)
    log.write(rows)
    log.close()
