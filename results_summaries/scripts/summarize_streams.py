"""Summarise every task-stream run on the cluster into two CSVs.

Most stream families have no written result, so the numbers quoted in
`results_summaries/*.md` come from here. The script reads only what the runs
already wrote -- `args.json` and `task_*/summary.json` for `run_tasks_stream`
runs, `config.json` and `task_end_metrics.csv` for `run_factor_rule_stream`
runs -- and trains nothing.

    ~/notebooks/.venv/bin/python results_summaries/scripts/summarize_streams.py

(it needs pandas, which is not a dependency of `plasticity/`; any Python with
pandas and numpy works, e.g. the notebooks project's venv)

It searches the same three places `visualization_scripts/common.results_dir`
does (`plasticity/results`, `$HPCVAULT/plasticity_results`,
`$WORK/plasticity_results`); `plasticity/results` also reaches the /hnvme
width sweep through its symlink. The packed MPI3D streams (`*.tar` in the
vault) are not read.

Outputs, under `results_summaries/csv/`:

    stream_runs.csv     one row per run (a stream, or a single-task baseline)
    stream_groups.csv   one row per (group, model), seeds pooled

Recurring tasks are compared the way the 2026-10-28 evidence audit did: for
each factor (or attribute) the mean test accuracy over its first five and its
last five occurrences, then averaged over factors. `train_acc` is the value the
runner logs, which is accumulated *during* the last epoch, not a final-model
evaluation.
"""
from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_summaries" / "csv"
SEARCH = [
    ROOT / "plasticity" / "results",
    Path(os.environ.get("HPCVAULT", "/home/vault/v170be/v170be10")) / "plasticity_results",
    Path(os.environ.get("WORK", "/home/atuin/v170be/v170be10")) / "plasticity_results",
]
K = 5  # occurrences per factor at each end of the stream


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 -- a half-written task is skipped, not fatal
        return None


def _dead(health: dict | None, layer: int):
    try:
        return health["layers"][layer]["dead"]
    except (TypeError, KeyError, IndexError):
        return None


def _ends(df: pd.DataFrame, key: str, value: str) -> tuple[float, float]:
    first, last = [], []
    for _, g in df.sort_values("task").groupby(key):
        first.append(g[value].head(K).mean())
        last.append(g[value].tail(K).mean())
    return float(np.mean(first)), float(np.mean(last))


def stream_run(group: str, run: Path) -> dict | None:
    args = _load(run / "args.json") or {}
    tasks = sorted(run.glob("task_*/summary.json"))
    rows = []
    for t in tasks:
        s = _load(t)
        if not s or "final_test_acc" not in s:
            continue
        rows.append({
            "task": s.get("task", int(t.parent.name.split("_")[1])),
            "factor": s.get("attribute", s.get("target_factor")),
            "test_acc": s["final_test_acc"],
            "train_acc": s.get("final_train_acc"),
            "dead_l1": _dead(s.get("health"), 0),
            "dead_l2": _dead(s.get("health"), 1),
            "units": (s.get("health") or {}).get("layers", [{}])[0].get("units"),
            "eff_rank": (s.get("health") or {}).get("effective_rank"),
        })
    if not rows:
        return None
    d = pd.DataFrame(rows).sort_values("task")
    # streams with one fixed target (permuted pixels, learnability) log no factor;
    # astype(str) would make that a missing key, which groupby silently drops
    d["factor"] = ["all" if v is None else str(v) for v in d["factor"]]
    acc_first, acc_last = _ends(d, "factor", "test_acc")
    tr_first, tr_last = _ends(d.dropna(subset=["train_acc"]), "factor", "train_acc")
    last = d.iloc[-1]
    return {
        "group": group, "run": run.name, "kind": "single" if len(d) == 1 and args.get("n_tasks", 1) <= 1 else "stream",
        "dataset": args.get("dataset_kind"), "model": args.get("model_kind"),
        "activation": args.get("activation"), "lr": args.get("learning_rate"),
        "epochs": args.get("num_epochs"), "center": args.get("center_inputs"),
        "offset": args.get("input_offset"), "perm": args.get("perm_percentage"),
        "seed": args.get("seed"), "n_tasks_cfg": args.get("n_tasks"), "n_tasks_done": len(d),
        "n_factors": d["factor"].nunique(),
        "test_acc_first5": acc_first, "test_acc_last5": acc_last,
        "train_acc_first5": tr_first, "train_acc_last5": tr_last,
        "units_per_layer": last["units"],
        "dead_l1_first": d["dead_l1"].iloc[0], "dead_l1_last": last["dead_l1"],
        "dead_l2_first": d["dead_l2"].iloc[0], "dead_l2_last": last["dead_l2"],
        "eff_rank_first": d["eff_rank"].iloc[0], "eff_rank_last": last["eff_rank"],
        "path": str(run),
    }


def factor_rule_run(group: str, run: Path) -> dict | None:
    cfg = _load(run / "config.json") or {}
    with (run / "task_end_metrics.csv").open() as f:
        d = pd.DataFrame(list(csv.DictReader(f)))
    if d.empty:
        return None
    d = d.rename(columns={"task_id": "task", "relevant_factor": "factor",
                          "current_task_test_accuracy": "test_acc"})
    d["task"] = d["task"].astype(int)
    d["test_acc"] = d["test_acc"].astype(float)
    acc_first, acc_last = _ends(d, "factor", "test_acc")
    return {
        "group": group, "run": run.name, "kind": "factor_rule", "dataset": cfg.get("dataset_kind"),
        "model": cfg.get("model_kind"), "activation": cfg.get("activation"), "lr": cfg.get("learning_rate"),
        "epochs": cfg.get("num_epochs"), "steps_per_task": cfg.get("steps_per_task"),
        "unfreeze_layers": cfg.get("unfreeze_layers"), "seed": cfg.get("seed"),
        "n_tasks_done": len(d), "n_factors": d["factor"].nunique(),
        "test_acc_first5": acc_first, "test_acc_last5": acc_last, "path": str(run),
    }


def find_runs() -> list[tuple[str, Path, str]]:
    found, seen = [], set()
    for root in SEARCH:
        if not root.is_dir():
            continue
        for gdir in sorted(p for p in root.iterdir() if p.is_dir()):
            # a group holds runs directly, or one more level down (reprdist pairs/triples, width sweep)
            for cand in [gdir, *gdir.iterdir(), *(q for p in gdir.iterdir() if p.is_dir() for q in p.iterdir())]:
                if not cand.is_dir():
                    continue
                real = cand.resolve()
                if real in seen:
                    continue
                if (cand / "task_000" / "summary.json").exists():
                    found.append((gdir.name, cand, "stream")); seen.add(real)
                elif (cand / "task_end_metrics.csv").exists():
                    found.append((gdir.name, cand, "factor_rule")); seen.add(real)
    return found


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = find_runs()
    with ThreadPoolExecutor(16) as pool:
        out = list(pool.map(lambda r: (stream_run if r[2] == "stream" else factor_rule_run)(r[0], r[1]), runs))
    df = pd.DataFrame([r for r in out if r])
    df.to_csv(OUT / "stream_runs.csv", index=False)
    agg = (df.groupby(["group", "model", "kind"], dropna=False)
           .agg(n_runs=("run", "size"),
                tasks_done_min=("n_tasks_done", "min"), tasks_done_max=("n_tasks_done", "max"),
                test_acc_first5=("test_acc_first5", "mean"), test_acc_last5=("test_acc_last5", "mean"),
                train_acc_first5=("train_acc_first5", "mean"), train_acc_last5=("train_acc_last5", "mean"),
                dead_l1_last=("dead_l1_last", "mean"), dead_l2_last=("dead_l2_last", "mean"),
                units_per_layer=("units_per_layer", "max"),
                eff_rank_first=("eff_rank_first", "mean"), eff_rank_last=("eff_rank_last", "mean"))
           .reset_index())
    agg["test_acc_change"] = agg["test_acc_last5"] - agg["test_acc_first5"]
    agg.to_csv(OUT / "stream_groups.csv", index=False)
    print(f"{len(df)} runs in {agg.shape[0]} groups -> {OUT}")


if __name__ == "__main__":
    main()
