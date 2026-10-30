"""
Summaries of a trained seed, the aggregate over the seeds of a run folder, saving and loading.

On disk: results/orthogonal_tasks/<run name>/
    seed<s>/config.json     the resolved config
    seed<s>/arrays.npz      the logged arrays (see train.py)
    seed<s>/summary.json    per unit and per phase summaries
    aggregate.json          totals and per-phase gaps over every seed in the folder (no selection)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from orthogonal_tasks.config import Config, task_name

RESULTS = Path(__file__).resolve().parent.parent / "results" / "orthogonal_tasks"


def _f(x):
    """JSON-safe float (None for NaN)."""
    x = float(x)
    return None if np.isnan(x) else x


def live_count(act_frac_train: np.ndarray) -> np.ndarray:
    """Units active on at least one train input, per logged row."""
    return (np.asarray(act_frac_train) > 0).sum(-1)


def steps_to(acc, steps, start, threshold):
    """Steps after `start` until the logged accuracy first reaches the threshold (None if never)."""
    hit = np.flatnonzero(acc >= threshold)
    return int(steps[hit[0]] - start) if len(hit) else None


def summarize(cfg: Config, arr: dict) -> dict:
    schedule, S, H = cfg.schedule, cfg.steps_per_task, cfg.width
    steps, death = arr["step"], arr["death_step"]

    def phase_of(step):
        return min(int(step - 1), len(schedule) * S - 1) // S        # the update at step - 1 did it

    units = []
    for h in range(H):
        d = None
        if death[h] >= 0:
            p = phase_of(death[h])
            d = {"step": int(death[h]), "phase": p, "task": int(schedule[p]),
                 "steps_after_switch": int(death[h] - p * S)}
        units.append({"unit": h, "death": d, "alive_at_end": bool(arr["act_frac_train"][-1, h] > 0),
                      "readout_init": _f(arr["a"][0, h]), "readout_final": _f(arr["a"][-1, h]),
                      "norm_init": _f(np.linalg.norm(arr["W"][0, h])),
                      "norm_final": _f(np.linalg.norm(arr["W"][-1, h]))})

    phases = []
    for p, task in enumerate(schedule):
        c_rows = np.flatnonzero((steps >= p * S) & (steps <= (p + 1) * S))
        c_start, c_end = c_rows[0], c_rows[-1]
        f_rows = np.flatnonzero(arr["fresh_phase"] == p)
        f_acc = arr["fresh_test_acc"][f_rows, task]
        c_acc = arr["test_acc"][c_end, task]
        fd = arr["fresh_death_step"][p]
        phases.append({
            "phase": p, "task": int(task),
            "continued_final_test_acc": _f(c_acc), "fresh_final_test_acc": _f(f_acc[-1]),
            "fresh_minus_continued_acc": _f(f_acc[-1] - c_acc),
            "fresh_solved": bool(f_acc[-1] >= cfg.solve_threshold),
            "continued_steps_to_threshold": steps_to(arr["test_acc"][c_rows, task], steps[c_rows], p * S,
                                                     cfg.solve_threshold),
            "fresh_steps_to_threshold": steps_to(f_acc, arr["fresh_step"][f_rows], p * S, cfg.solve_threshold),
            "continued_live_start": int(live_count(arr["act_frac_train"][c_start])),
            "continued_live_end": int(live_count(arr["act_frac_train"][c_end])),
            "fresh_live_end": int(live_count(arr["fresh_act_frac_train"][f_rows[-1]])),
            "continued_deaths": int(sum(p * S < s <= (p + 1) * S for s in death if s >= 0)),
            "fresh_deaths": int((fd >= 0).sum()),
        })

    return {"seed": cfg.seed, "config": asdict(cfg), "units": units, "phases": phases,
            "died": int((death >= 0).sum()), "alive_at_end": int(live_count(arr["act_frac_train"][-1])),
            "fresh_solved_every_phase": all(ph["fresh_solved"] for ph in phases)}


def save_seed(run_dir: Path, cfg: Config, arrays: dict, summary: dict) -> Path:
    sd = run_dir / f"seed{cfg.seed}"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    np.savez_compressed(sd / "arrays.npz", **arrays)
    (sd / "summary.json").write_text(json.dumps(summary, indent=2))
    return sd


def seeds_in(run_dir: Path) -> list[int]:
    return sorted(int(p.parent.name[4:]) for p in Path(run_dir).glob("seed*/summary.json"))


def load_seed(run_dir: Path, seed: int) -> tuple[Config, dict, dict]:
    """(config, summary, arrays) of one saved seed."""
    sd = Path(run_dir) / f"seed{seed}"
    cfg = Config(**json.loads((sd / "config.json").read_text()))
    return cfg, json.loads((sd / "summary.json").read_text()), dict(np.load(sd / "arrays.npz"))


def aggregate(run_dir: Path) -> dict:
    """Totals over every seed of a run folder and the per-phase fresh − continued gaps."""
    summaries = [json.loads(p.read_text()) for p in sorted(Path(run_dir).glob("seed*/summary.json"))]
    gaps = np.array([[ph["fresh_minus_continued_acc"] for ph in s["phases"]] for s in summaries], dtype=float)
    solved = np.array([[ph["fresh_solved"] for ph in s["phases"]] for s in summaries], dtype=bool)
    with warnings.catch_warnings():         # a phase no fresh net solved has no gap to average
        warnings.simplefilter("ignore", RuntimeWarning)
        where_solved = np.nanmean(np.where(solved, gaps, np.nan), 0)
    out = {
        "seeds": [s["seed"] for s in summaries],
        "units": sum(len(s["units"]) for s in summaries),
        "died": sum(s["died"] for s in summaries),
        "alive_at_end": sum(s["alive_at_end"] for s in summaries),
        "deaths_per_phase": np.sum([[ph["continued_deaths"] for ph in s["phases"]] for s in summaries], 0).tolist(),
        "fresh_deaths_per_phase": np.sum([[ph["fresh_deaths"] for ph in s["phases"]] for s in summaries],
                                         0).tolist(),
        "gap_mean_per_phase": np.mean(gaps, 0).tolist(),
        "gap_mean_where_fresh_solved": [None if np.isnan(v) else float(v) for v in where_solved],
        "fresh_solved_per_phase": solved.sum(0).tolist(),
    }
    (Path(run_dir) / "aggregate.json").write_text(json.dumps(out, indent=2))
    return out


def print_seed_report(summary: dict) -> None:
    sched = [ph["task"] for ph in summary["phases"]]
    print(f"  died {summary['died']} of {len(summary['units'])} units, alive at end {summary['alive_at_end']}; "
          f"fresh nets solve every phase: {summary['fresh_solved_every_phase']}")
    for u in summary["units"]:
        d = u["death"]
        fate = (f"died at step {d['step']} (phase {d['phase']}, task {task_name(d['task'])}, "
                f"+{d['steps_after_switch']} after the switch)") if d else "alive"
        print(f"    unit {u['unit']}: a {u['readout_init']:+.2f} -> {u['readout_final']:+.2f}, "
              f"|w| {u['norm_init']:.2f} -> {u['norm_final']:.2f}; {fate}")
    print("    final acc per phase, continued / fresh: " + "  ".join(
        f"{task_name(t)} {ph['continued_final_test_acc']:.2f}/{ph['fresh_final_test_acc']:.2f}"
        for t, ph in zip(sched, summary["phases"])))


def print_aggregate(agg: dict) -> None:
    print(f"\nAll seeds {agg['seeds']}: died {agg['died']} of {agg['units']} units, alive at end {agg['alive_at_end']}")
    print("  deaths per phase, continued: " + " ".join(map(str, agg["deaths_per_phase"])))
    print("  deaths per phase, fresh:     " + " ".join(map(str, agg["fresh_deaths_per_phase"])))
    print("  mean fresh − continued acc per phase: " + " ".join(f"{v:+.2f}" for v in agg["gap_mean_per_phase"]))
    print("  same, where the fresh net solved:     " + " ".join(
        "  n/a" if v is None else f"{v:+.2f}" for v in agg["gap_mean_where_fresh_solved"])
          + f"  (seeds per phase {agg['fresh_solved_per_phase']})")
