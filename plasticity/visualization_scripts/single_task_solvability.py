"""
visualization_scripts/single_task_solvability.py — Is every task of a factor stream learnable
alone? Reads the single-task runs (submit_stream_array.sh SINGLE=1, e.g. launch_robot3d_single.sh:
each factor trained on a fresh network) and prints one row per (run folder, factor), averaged
over seeds:

    factor, label rule, class balance (train share of label 1) and the majority-class accuracy,
    test accuracy after --budget epochs (the stream's per-task budget) and at the end,
    final train accuracy, and a verdict:
        solved    final test accuracy >= --solved
        learns    above the majority class by more than --margin, but not solved
        chance    within --margin of the majority class

Run with:
    uv run python -m visualization_scripts.single_task_solvability \
        --runs results/mpi3d_realistic_factor_single_mlp_2x30 results/isaac3d_factor_single_mlp_2x30
    uv run python -m visualization_scripts.single_task_solvability --runs <folder> --budget 10 --solved 0.9

Prints only; writes nothing.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.download import SHAPES3D_FACTOR_NAMES
from data.robot3d import ROBOT3D


def factor_name(dataset_kind, factor):
    names = ROBOT3D[dataset_kind].factor_names if dataset_kind in ROBOT3D else SHAPES3D_FACTOR_NAMES
    return names[factor] if factor < len(names) else str(factor)


def read_runs(folder: Path, budget: int):
    """{factor: [one dict per seed]} of the single-task run directories in `folder`."""
    rows = defaultdict(list)
    for summary_path in sorted(folder.glob("stream_*_seed*/summary.json")):
        run = summary_path.parent
        args = json.loads((run / "args.json").read_text())
        tasks = json.loads(summary_path.read_text())["tasks"]
        if not tasks:
            continue                                         # not finished yet
        if len(tasks) != 1:
            print(f"skipping {run.name}: {len(tasks)} tasks, not a single-task run")
            continue
        task = tasks[0]
        history = json.loads((run / "task_000" / "history.json").read_text())
        test_acc = history["test_acc"]
        balance = task.get("class_balance")
        rows[task["factors"][0]].append(dict(
            dataset=args["dataset_kind"], rule=task.get("rule", args.get("label_rule") or "parity"),
            balance=balance, majority=None if balance is None else max(balance, 1 - balance),
            at_budget=test_acc[min(budget, len(test_acc)) - 1], final=test_acc[-1],
            train=history["train_acc"][-1], epochs=len(test_acc)))
    return rows


def verdict(final, majority, solved, margin):
    if final >= solved:
        return "solved"
    if majority is not None and final - majority <= margin:
        return "chance"
    return "learns"


def mean_std(values):
    return f"{np.mean(values):.3f}±{np.std(values):.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", required=True, help="folders holding single-task run directories")
    parser.add_argument("--budget", type=int, default=10, help="epochs a task gets in the stream")
    parser.add_argument("--solved", type=float, default=0.95, help="final test accuracy that counts as solved")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="above the majority class by at most this much counts as chance")
    args = parser.parse_args(argv)

    header = (f"{'factor':<22}{'rule':<12}{'balance':>8}{'majority':>9}{'seeds':>6}"
              f"{f'test@{args.budget}':>14}{'test final':>14}{'train final':>14}  verdict")
    for folder in map(Path, args.runs):
        rows = read_runs(folder, args.budget)
        print(f"\n{folder}")
        if not rows:
            print("  no finished single-task runs")
            continue
        print(header)
        for factor in sorted(rows):
            seeds = rows[factor]
            first = seeds[0]
            majority = None if first["majority"] is None else float(np.mean([s["majority"] for s in seeds]))
            final = [s["final"] for s in seeds]
            balance = "" if first["balance"] is None else "%.3f" % np.mean([s["balance"] for s in seeds])
            majority_str = "" if majority is None else "%.3f" % majority
            short = "" if first["epochs"] >= args.budget else " (only %d epochs)" % first["epochs"]
            print(f"{factor_name(first['dataset'], factor):<22}{first['rule']:<12}"
                  f"{balance:>8}{majority_str:>9}{len(seeds):>6}"
                  f"{mean_std([s['at_budget'] for s in seeds]):>14}{mean_std(final):>14}"
                  f"{mean_std([s['train'] for s in seeds]):>14}  "
                  f"{verdict(float(np.mean(final)), majority, args.solved, args.margin)}{short}")


if __name__ == "__main__":
    main()
