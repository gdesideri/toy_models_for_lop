"""
orthogonal_tasks/run.py — train a run (several seeds) of the orthogonal-tasks toy and save it.

Usage (from plasticity/)
    uv run python -m orthogonal_tasks.run --seeds 0 --steps-per-task 500 --phases 3          # smoke test
    uv run python -m orthogonal_tasks.run --seeds 0 1 2                                      # 2-D half-planes
    uv run python -m orthogonal_tasks.run --label-rule stripe --seeds 0 1 2                  # stripes
    uv run python -m orthogonal_tasks.run --label-rule xor --seeds 0 1 2                     # XOR
    uv run python -m orthogonal_tasks.run --input-dim 3 --task-normals diagonal --scales 0.5 1 1 --seeds 0 1 2
    uv run python -m orthogonal_tasks.run --input-dim 3 --n-tasks 3 --task-normals diagonal-c --seeds 0 1 2
then
    uv run python -m visualization_scripts.toy_models.animate_orthogonal_tasks --run-dir <printed dir> --seed 0
    uv run python -m visualization_scripts.toy_models.plot_orthogonal_summary --run-dirs <dir> ... --out <png>

Defaults: K = 2 tasks on the coordinate axes, D = 2, cloud half-widths (1, 0.5), 4 units, random hidden-bias
init, plain SGD lr 0.1, batch 64, 10 000 steps per task, 11 phases (10 switches). See data.py, network.py
and train.py for the setup and what is logged, summary.py for the on-disk layout.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from orthogonal_tasks.config import BIAS_INITS, Config, run_name
from orthogonal_tasks.data import LABEL_RULES, NORMAL_KINDS
from orthogonal_tasks.summary import RESULTS, aggregate, print_aggregate, print_seed_report, save_seed, summarize
from orthogonal_tasks.train import run_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    p.add_argument("--n-tasks", type=int, default=d.n_tasks)
    p.add_argument("--input-dim", type=int, default=None, help="D (default max(2, K))")
    p.add_argument("--scales", type=float, nargs="+", default=None,
                   help="half-widths of the cloud per axis, D values (default 1, ..., 1, 0.5)")
    p.add_argument("--label-rule", choices=LABEL_RULES, default=d.label_rule)
    p.add_argument("--task-normals", choices=NORMAL_KINDS, default=d.task_normals)
    p.add_argument("--width", type=int, default=d.width)
    p.add_argument("--bias-init", choices=BIAS_INITS, default=d.bias_init)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default=d.optimizer)
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--steps-per-task", type=int, default=d.steps_per_task)
    p.add_argument("--phases", type=int, default=d.phases, help="phases in total (switches = phases - 1)")
    p.add_argument("--n-train", type=int, default=d.n_train)
    p.add_argument("--n-test", type=int, default=d.n_test)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--solve-threshold", type=float, default=d.solve_threshold)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", type=Path, default=RESULTS, help="results root")
    p.add_argument("--tag", default="", help="suffix of the run folder name")
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    fields = {k: v for k, v in vars(args).items() if k not in ("seeds", "out", "tag")}
    base = Config(**fields).resolved()
    run_dir = args.out / (run_name(base) + (f"_{args.tag}" if args.tag else ""))
    print(f"Orthogonal tasks: K={base.n_tasks} D={base.input_dim} scales={base.scales} width={base.width} "
          f"rule={base.label_rule} schedule={base.schedule} -> {run_dir}")
    for seed in args.seeds:
        t0 = time.time()
        cfg = Config(**{**asdict(base), "seed": seed})
        arrays = run_seed(cfg)
        summary = summarize(cfg, arrays)
        save_seed(run_dir, cfg, arrays, summary)
        print(f"seed {seed} ({time.time() - t0:.1f}s)")
        print_seed_report(summary)
    print_aggregate(aggregate(run_dir))
    print(f"\nRun dir: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
