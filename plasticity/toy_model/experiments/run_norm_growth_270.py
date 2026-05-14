"""
run_norm_growth_270.py — Align to w0, grow norms, then rotate 270° to w3=(0,-1).

Three phases:
  Phase 1  N_ALIGN=10 000 steps,  y = w0·x           (align units to w0=(1,0))
  Phase 2  N_NORM =5 000  steps,  y = SCALE * w0·x   (grow norms ×SCALE)
  Phase 3  N_TEST steps,          y = w3·x            (w3=(0,-1), 270° from w0)

Key difference from run_norm_growth.py (180° flip):
  The 180° case creates a COMPLETE dead zone: units aligned to w0=(1,0)
  receive zero gradient for inputs in the x1<0 half-space, which is exactly
  where w2=(-1,0) lives. No rotation is possible.

  The 270° case (target w3=(0,-1)) has NO complete dead zone:
  units at 0°=(1,0) activate for x1>0; the target signal is x2, which is
  independent of x1. Both half-spaces provide gradient about x2.
  → units should actually rotate from 0° toward -90° (270°).

Also saves fc2 weights in every snapshot so we can visualise the signed
effective contribution fc2[u]*fc1[u] of each unit.

Run with:  uv run python -m experiments.run_norm_growth_270
"""
from __future__ import annotations

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.benchmark import make_rng

# ── Config ────────────────────────────────────────────────────────────────────

K      = 2
HIDDEN = 4
BATCH  = 512
LR     = 1e-3
N_SEEDS = 5
SNAPSHOT_EVERY = 10

N_ALIGN    = 10_000
SCALE      = 20
N_NORM     = 5_000
N_TEST_ALL = [2_000, 5_000, 7_000]

W0 = torch.tensor([ 1.0,  0.0])   # 0°
W3 = torch.tensor([ 0.0, -1.0])   # 270°  ← target direction

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "norm_growth_270")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Network ───────────────────────────────────────────────────────────────────

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(K, HIDDEN, bias=False)
        self.fc2 = nn.Linear(HIDDEN, 1,  bias=False)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def fresh_net():
    return Net()


# ── Helpers ───────────────────────────────────────────────────────────────────

def mse(pred, y):
    return F.mse_loss(pred, y)

def angle_deg(vec, ref):
    u = vec / (vec.norm() + 1e-10)
    return math.degrees(math.acos(float(torch.clamp(u @ ref, -1, 1).abs())))

def _to_json(obj):
    if isinstance(obj, torch.Tensor):   return obj.tolist()
    if isinstance(obj, dict):           return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):  return [_to_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj

def save_results(name, data):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(_to_json(data), f, indent=2)
    print(f"  → saved {path}")


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(seed: int, n_test: int) -> dict:
    torch.manual_seed(seed)
    rng = make_rng(seed)

    model = fresh_net()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    def snapshot(step, phase, loss_val):
        w1 = model.fc1.weight.detach().clone()          # (HIDDEN, K)
        w2 = model.fc2.weight.detach().squeeze().clone() # (HIDDEN,)
        norms  = w1.norm(dim=1).tolist()
        angles_to_w0 = [angle_deg(w1[i], W0) for i in range(HIDDEN)]
        # signed atan2 direction of each fc1 unit
        dir_angles = [math.degrees(math.atan2(float(w1[i, 1]), float(w1[i, 0])))
                      for i in range(HIDDEN)]
        # effective contribution vector: fc2[u] * fc1[u]  (K-dim)
        eff_vecs = (w2.unsqueeze(1) * w1).tolist()  # (HIDDEN, K)
        return {
            "step":       step,
            "phase":      phase,
            "loss":       loss_val,
            "fc1_weight": w1,
            "fc2_weight": w2,
            "fc1_norms":  norms,
            "fc1_angles_to_w0": angles_to_w0,
            "fc1_dir_angles":   dir_angles,
            "eff_vecs":   eff_vecs,   # signed effective direction of each unit
        }

    snapshots = [snapshot(0, "init", None)]
    global_step = 0

    # ── Phase 1: align to w0 ───────────────────────────────────────────────

    phase1_curve = []
    for s in range(N_ALIGN):
        x = torch.randn(BATCH, K, generator=rng)
        y = x @ W0
        loss = mse(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        global_step += 1
        if global_step % SNAPSHOT_EVERY == 0:
            snapshots.append(snapshot(global_step, "align", loss.item()))
        if s % 100 == 0:
            phase1_curve.append((global_step, loss.item()))

    w_end1 = model.fc1.weight.detach()
    norms_end1 = w_end1.norm(dim=1).tolist()
    print(f"    [phase1 end]  loss={phase1_curve[-1][1]:.6f}"
          f"  fc1 norms={[f'{n:.3f}' for n in norms_end1]}")

    # ── Phase 2: grow norms ────────────────────────────────────────────────

    phase2_curve = []
    for s in range(N_NORM):
        x = torch.randn(BATCH, K, generator=rng)
        y = SCALE * (x @ W0)
        loss = mse(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        global_step += 1
        if global_step % SNAPSHOT_EVERY == 0:
            snapshots.append(snapshot(global_step, "norm_grow", loss.item()))
        if s % 100 == 0:
            phase2_curve.append((global_step, loss.item()))

    w_end2 = model.fc1.weight.detach()
    norms_end2 = w_end2.norm(dim=1).tolist()
    dir2 = [math.degrees(math.atan2(float(w_end2[i,1]), float(w_end2[i,0])))
            for i in range(HIDDEN)]
    print(f"    [phase2 end]  loss={phase2_curve[-1][1]:.6f}"
          f"  fc1 norms={[f'{n:.3f}' for n in norms_end2]}"
          f"  dirs={[f'{a:.0f}' for a in dir2]}°")

    # ── Phase 3: w3=(0,-1) — prefix vs fresh ──────────────────────────────

    f_model = fresh_net()
    f_opt   = torch.optim.Adam(f_model.parameters(), lr=LR)

    prefix_curve, fresh_curve = [], []

    log_every = max(1, n_test // 100)
    for s in range(n_test):
        x = torch.randn(BATCH, K, generator=rng)
        y = x @ W3                    # 270° target

        loss   = mse(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

        f_loss = mse(f_model(x), y)
        f_opt.zero_grad(); f_loss.backward(); f_opt.step()

        global_step += 1
        if global_step % SNAPSHOT_EVERY == 0:
            snapshots.append(snapshot(global_step, "test", loss.item()))
        if s % log_every == 0:
            prefix_curve.append((global_step, loss.item()))
            fresh_curve.append((global_step, f_loss.item()))

    lop_gap = prefix_curve[-1][1] - fresh_curve[-1][1]
    w_end3 = model.fc1.weight.detach()
    norms_end3 = w_end3.norm(dim=1).tolist()
    dir3 = [math.degrees(math.atan2(float(w_end3[i,1]), float(w_end3[i,0])))
            for i in range(HIDDEN)]
    print(f"    [phase3 end]  loss={prefix_curve[-1][1]:.6f}"
          f"  fresh={fresh_curve[-1][1]:.6f}  LoP={lop_gap:+.6f}"
          f"  fc1 norms={[f'{n:.3f}' for n in norms_end3]}"
          f"  dirs={[f'{a:.0f}' for a in dir3]}°")

    return {
        "seed":    seed,
        "n_test":  n_test,
        "snapshots":     snapshots,
        "phase1_curve":  phase1_curve,
        "phase2_curve":  phase2_curve,
        "prefix_curve":  prefix_curve,
        "fresh_curve":   fresh_curve,
        "lop_gap":       lop_gap,
        "norms_end_phase1": norms_end1,
        "norms_end_phase2": norms_end2,
        "norms_end_phase3": norms_end3,
    }


def run_experiment():
    print(f"\n═══ Norm growth 270° experiment  (w0→w3, NO dead zone) ═══")
    print(f"  phase1={N_ALIGN} (align to w0=(1,0))  "
          f"phase2={N_NORM} (scale×{SCALE})  "
          f"phase3 target=w3=(0,-1)  sweep={N_TEST_ALL}")

    results = {}
    for n_test in N_TEST_ALL:
        print(f"\n  ── n_test={n_test} ──")
        seed_results = [run_one(s, n_test) for s in range(N_SEEDS)]
        mean_lop = sum(r["lop_gap"] for r in seed_results) / N_SEEDS
        print(f"  mean LoP = {mean_lop:+.6f}")
        key = f"ntest{n_test}"
        results[key] = {
            "seeds":    seed_results,
            "mean_lop": mean_lop,
            "n_test":   n_test,
        }

    save_results("exp_norm_growth_270", {
        "conditions": results,
        "config": {
            "k": K, "hidden": HIDDEN, "lr": LR, "batch": BATCH,
            "n_align": N_ALIGN, "scale": SCALE, "n_norm": N_NORM,
            "n_test_all": N_TEST_ALL,
            "target_angle": 270,
            "w_target": W3.tolist(),
        },
    })


if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    print(f"\nDone in {time.time()-t0:.1f}s")
