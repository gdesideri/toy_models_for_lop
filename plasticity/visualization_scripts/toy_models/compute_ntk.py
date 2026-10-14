"""
compute_ntk.py — Compute the empirical NTK for a model at a given training snapshot.

Loads model weights from an experiment JSON (rotating_target or norm_growth_270),
samples N_SAMPLES points uniformly from the data distribution, and computes:

  - Full NTK matrix K  (N, N)
  - Per-layer NTK matrices K_fc1, K_fc2  (N, N) each
  - Full Jacobian J  (N, P)

Saves results to results/ntk/<experiment>_<condition>_step<S>.json

Config
──────
Set EXPERIMENT, CONDITION, SNAPSHOT_STEP and TASK_ANGLE below.

Run with:  uv run python visualization_scripts/toy_models/compute_ntk.py
"""
import json, os, sys, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.toy import MinimalNet
from src.ntk    import (compute_ntk, compute_ntk_per_layer,
                         compute_jacobian, compute_jacobian_per_layer)

# ── Config ────────────────────────────────────────────────────────────────────

# Which experiment to pull the model from
EXPERIMENT  = "norm_growth_270"          # "rotating_target" | "norm_growth_270"
CONDITION   = "ntest5000"               # e.g. "steps10000_cycles1" | "ntest5000"
SEED_IDX    = 0
SNAPSHOT_STEP = None   # None → use last snapshot; or set to a global step number

# Data: sample N_SAMPLES points from the data distribution
N_SAMPLES   = 128
TASK_ANGLE  = 0        # degrees — which task's target direction to use for residuals
K           = 2        # input dimension (fixed for all our experiments)

RESULTS     = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
OUT_DIR     = os.path.join(RESULTS, "ntk")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load experiment JSON ───────────────────────────────────────────────────────

exp_path = os.path.join(RESULTS, EXPERIMENT, f"exp_{EXPERIMENT}.json")
with open(exp_path) as f:
    exp = json.load(f)

# Navigate to the right condition and seed
if EXPERIMENT == "rotating_target":
    seed_data = exp["steps_sweep"][CONDITION]["seeds"][SEED_IDX]
elif EXPERIMENT == "norm_growth_270":
    seed_data = exp["conditions"][CONDITION]["seeds"][SEED_IDX]
else:
    raise ValueError(f"Unknown experiment: {EXPERIMENT}")

# Find snapshot
snaps = [s for s in seed_data["snapshots"] if s.get("fc1_weight") is not None]
if SNAPSHOT_STEP is None:
    snap = snaps[-1]
else:
    snap = min(snaps, key=lambda s: abs(s["step"] - SNAPSHOT_STEP))

step = snap["step"]
print(f"Using snapshot at step {step}")

# ── Reconstruct model ──────────────────────────────────────────────────────────

model = MinimalNet(d_in=K, hidden=4, bias=False)

w1 = torch.tensor(snap["fc1_weight"], dtype=torch.float32)   # (4, 2)
model.fc1.weight.data.copy_(w1)

if "fc2_weight" in snap:
    w2 = torch.tensor(snap["fc2_weight"], dtype=torch.float32)
    model.fc2.weight.data.copy_(w2.reshape(1, -1))
    print("  fc2 loaded from snapshot")
else:
    print("  WARNING: fc2_weight not in snapshot — fc2 kept at random init")
    w2 = model.fc2.weight.data.squeeze()

model.eval()

# ── Sample data ────────────────────────────────────────────────────────────────

torch.manual_seed(42)
x = torch.randn(N_SAMPLES, K)

rad    = math.radians(TASK_ANGLE)
w_task = torch.tensor([math.cos(rad), math.sin(rad)])
y      = x @ w_task     # (N,)

# ── Compute NTK ───────────────────────────────────────────────────────────────

print(f"Computing full NTK ({N_SAMPLES}×{N_SAMPLES})...")
K_full = compute_ntk(model, x)                  # (N, N)

print("Computing per-layer NTKs...")
K_layers = compute_ntk_per_layer(model, x)      # dict: name → (N, N)

print("Computing full Jacobian...")
J_full   = compute_jacobian(model, x)           # (N, P)

# ── Residuals ──────────────────────────────────────────────────────────────────

with torch.no_grad():
    f_x = model(x)   # (N,)
residuals = y - f_x   # (N,)

# ── Save ───────────────────────────────────────────────────────────────────────

tag = f"{EXPERIMENT}_{CONDITION}_s{SEED_IDX}_step{step}_N{N_SAMPLES}_task{TASK_ANGLE}"

out = {
    "experiment":   EXPERIMENT,
    "condition":    CONDITION,
    "seed_idx":     SEED_IDX,
    "snapshot_step": step,
    "n_samples":    N_SAMPLES,
    "task_angle":   TASK_ANGLE,
    "K_full":       K_full.tolist(),
    "K_layers":     {name: K.tolist() for name, K in K_layers.items()},
    "J_full":       J_full.tolist(),
    "x":            x.tolist(),
    "y":            y.tolist(),
    "f_x":          f_x.tolist(),
    "residuals":    residuals.tolist(),
    "fc1_weight":   w1.tolist(),
    "fc2_weight":   w2.tolist(),
}

out_path = os.path.join(OUT_DIR, f"{tag}.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved → {out_path}")

# ── Quick summary ──────────────────────────────────────────────────────────────

K_np    = K_full.numpy()
eigvals = np.linalg.eigvalsh(K_np)[::-1]   # descending
print(f"\nNTK summary:")
print(f"  shape:       {K_np.shape}")
print(f"  trace:       {K_np.trace():.4f}")
print(f"  rank (ε=1e-4): {(eigvals > 1e-4 * eigvals[0]).sum()}")
print(f"  top-5 eigenvalues: {eigvals[:5].round(4)}")
print(f"  residual norm: {residuals.norm().item():.6f}")
