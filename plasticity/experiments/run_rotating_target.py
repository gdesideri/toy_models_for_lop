"""
run_rotating_target.py — Rotating target regression.

Setup
─────
  k = 2,  x ~ N(0, I_2)  (uniform variance, no suppressed dims)

  4 tasks with target weight rotated 90° each step:
      w_0 = ( 1,  0)   →   0°
      w_1 = ( 0,  1)   →  90°
      w_2 = (-1,  0)   → 180°
      w_3 = ( 0, -1)   → 270°

  Target : y = w_i · x          (linear projection, MSE loss)
  Network: MinimalNet  k=2 → 4 hidden (ReLU, no bias) → 1 output

LoP
───
  At every task, a fresh model (same arch, re-initialised) trains in
  parallel on the SAME mini-batches as the prefix model.
  LoP_gap = MSE_final(prefix) − MSE_final(fresh)

Sweep
─────
  steps_per_task : [500, 2000, 5000]   n_cycles=1 each
  n_cycles       : [1, 3, 6]           steps=2000  each
  (allows plotting LoP as a function of both axes)

Snapshots: fc1.weight (4×2) every SNAPSHOT_EVERY steps → weight trajectories

Run with:  uv run python -m experiments.run_rotating_target
"""
from __future__ import annotations

import copy, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.toy_regression import make_rng
from models.factory import build_model

# ── Config ────────────────────────────────────────────────────────────────────

K             = 2
HIDDEN        = 4
N_SEEDS       = 5
BATCH         = 512
LR            = 1e-3
SNAPSHOT_EVERY = 10

STEPS_SWEEP  = [2000, 10000, 50000]   # n_cycles=1
CYCLES_SWEEP = [1, 3, 6]             # steps=10000

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "rotating_target")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Target weight vectors (90° rotations in 2D) ───────────────────────────────

TASK_ANGLES = [0, 90, 180, 270]          # degrees
TASK_WS = [
    torch.tensor([math.cos(math.radians(a)), math.sin(math.radians(a))])
    for a in TASK_ANGLES
]                                         # list of 4 unit vectors in R^2


# ── Data / loss ───────────────────────────────────────────────────────────────

def sample(w: torch.Tensor, n: int, rng: torch.Generator):
    x = torch.randn(n, K, generator=rng)
    y = x @ w                             # linear projection
    return x, y

def mse(pred, y):
    return F.mse_loss(pred, y)


# ── Fresh model ───────────────────────────────────────────────────────────────

def fresh_model() -> nn.Module:
    return build_model("minimal", K)      # 4 ReLU units, no bias


# ── Serialisation ─────────────────────────────────────────────────────────────

def _to_json(obj):
    if isinstance(obj, torch.Tensor):  return obj.tolist()
    if isinstance(obj, dict):          return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj

def save_results(name, data):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(_to_json(data), f, indent=2)
    print(f"  → saved {path}")


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(steps_per_task: int, n_cycles: int, seed: int) -> dict:
    torch.manual_seed(seed)
    rng = make_rng(seed)

    model = fresh_model()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    def snapshot(step, cycle, task_idx, loss_val):
        # fc1.weight: (H, k) — the feature directions
        fc1_w = None
        for name, p in model.named_parameters():
            if "fc1" in name and p.dim() == 2:
                fc1_w = p.detach().clone(); break
        return {"step": step, "cycle": cycle, "task_idx": task_idx,
                "fc1_weight": fc1_w, "loss": loss_val}

    snapshots   = [snapshot(0, -1, -1, None)]
    block_logs  = []
    global_step = 0
    n_tasks     = len(TASK_WS)

    for cycle in range(n_cycles):
        for task_idx, w in enumerate(TASK_WS):
            presentation = cycle * n_tasks + task_idx

            # fresh model for this task (same arch, new random init)
            f_model = fresh_model()
            f_opt   = torch.optim.Adam(f_model.parameters(), lr=LR)

            prefix_curve, fresh_curve = [], []

            for s in range(steps_per_task):
                x, y = sample(w, BATCH, rng)         # same batch both models

                # prefix
                loss   = mse(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()

                # fresh
                f_loss = mse(f_model(x), y)
                f_opt.zero_grad(); f_loss.backward(); f_opt.step()

                global_step += 1

                if global_step % SNAPSHOT_EVERY == 0:
                    snapshots.append(snapshot(global_step, cycle, task_idx, loss.item()))

                if s % 50 == 0:
                    prefix_curve.append((global_step, loss.item()))
                    fresh_curve.append( (global_step, f_loss.item()))

            lop_gap = prefix_curve[-1][1] - fresh_curve[-1][1]

            block_logs.append({
                "cycle":        cycle,
                "task_idx":     task_idx,
                "presentation": presentation,
                "angle_deg":    TASK_ANGLES[task_idx],
                "w":            w,
                "prefix_loss":  prefix_curve,
                "fresh_loss":   fresh_curve,
                "final_loss":   prefix_curve[-1][1],
                "lop_gap":      lop_gap,
                "lop_step":     global_step - steps_per_task,
            })
            print(f"      cy{cycle} t{task_idx} w={TASK_ANGLES[task_idx]:3d}°"
                  f"  loss={prefix_curve[-1][1]:.5f}  fresh={fresh_curve[-1][1]:.5f}"
                  f"  LoP={lop_gap:+.5f}  step={global_step}")

    snapshots.append(snapshot(global_step, n_cycles-1, n_tasks-1,
                              block_logs[-1]["final_loss"]))

    mean_lop = float(sum(b["lop_gap"] for b in block_logs) / len(block_logs))
    print(f"      mean LoP={mean_lop:+.5f}  n_snapshots={len(snapshots)}")

    return {
        "seed": seed,
        "steps_per_task": steps_per_task, "n_cycles": n_cycles,
        "snapshots":  snapshots,
        "block_logs": block_logs,
        "task_ws":    TASK_WS,
        "task_angles": TASK_ANGLES,
    }


def _mean_lop(seed_results):
    return float(
        sum(sum(b["lop_gap"] for b in r["block_logs"]) / len(r["block_logs"])
            for r in seed_results) / len(seed_results))


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment():
    results = {"steps_sweep": {}, "cycles_sweep": {}}

    print("\n═══ Sweep 1: steps_per_task (n_cycles=1) ═══")
    for steps in STEPS_SWEEP:
        print(f"\n  steps={steps}")
        seed_results = [run_one(steps, 1, s) for s in range(N_SEEDS)]
        key = f"steps{steps}_cycles1"
        ml  = _mean_lop(seed_results)
        results["steps_sweep"][key] = {
            "steps_per_task": steps, "n_cycles": 1,
            "seeds": seed_results, "mean_lop": ml,
        }
        print(f"  → mean LoP = {ml:+.5f}")

    print("\n═══ Sweep 2: n_cycles (steps=2000) ═══")
    for n_cycles in CYCLES_SWEEP:
        print(f"\n  n_cycles={n_cycles}")
        seed_results = [run_one(10000, n_cycles, s) for s in range(N_SEEDS)]
        key = f"steps10000_cycles{n_cycles}"
        ml  = _mean_lop(seed_results)
        results["cycles_sweep"][key] = {
            "steps_per_task": 10000, "n_cycles": n_cycles,
            "seeds": seed_results, "mean_lop": ml,
        }
        print(f"  → mean LoP = {ml:+.5f}")

    save_results("exp_rotating_target", results)
    return results


if __name__ == "__main__":
    t0 = time.time()
    print("Rotating target experiment")
    print(f"  k={K}, tasks={TASK_ANGLES}°, target=w·x, loss=MSE")
    run_experiment()
    print(f"\nDone in {time.time()-t0:.1f}s")
