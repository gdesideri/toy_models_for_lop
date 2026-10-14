"""
ntk_alignment.py — Residual alignment with NTK eigenvectors.

Loads:
  - A precomputed eigendecomposition (from ntk_eigenvalues.py)
  - The model at the same snapshot (weights from the NTK JSON)
  - The same data distribution

Computes for each eigenvector v_k:
    alignment_k = (v_k^T r)^2 / ||r||^2

where r = y - f(x) is the residual vector on the sampled batch.

Also computes the "learning speed" proxy:
    speed_k = λ_k * alignment_k

High speed_k → this component of the residual will be eliminated quickly.
Low speed_k (either small λ or small alignment) → slow learning / LoP.

Config
──────
EIGS_FILE : eigendecomposition JSON (from ntk_eigenvalues.py)
NTK_KEY   : which NTK to use: "full", "fc1.weight", "fc2.weight"
TASK_ANGLE: task direction for computing residuals (may differ from training)

Run with:  uv run python visualization_scripts/toy_models/ntk_alignment.py
"""
import json, os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ntk import residual_alignment

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS   = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
NTK_DIR   = os.path.join(RESULTS, "ntk")

EIGS_FILE  = None     # e.g. "eigs_rotating_target_...json" — None → auto last
NTK_KEY    = "full"   # "full" | "fc1.weight" | "fc2.weight"
TASK_ANGLE = None     # None → use angle from the NTK JSON

# ── Load eigendecomposition ───────────────────────────────────────────────────

if EIGS_FILE is None:
    files = sorted([f for f in os.listdir(NTK_DIR) if f.startswith("eigs_")])
    if not files:
        raise FileNotFoundError(f"No eigs_ files in {NTK_DIR}. Run ntk_eigenvalues.py first.")
    EIGS_FILE = files[-1]
    print(f"Auto-selected eigs file: {EIGS_FILE}")

eigs_path = os.path.join(NTK_DIR, EIGS_FILE)
with open(eigs_path) as f:
    eigs = json.load(f)

# Also load the original NTK file for x, y, weights
ntk_file = eigs["source_file"]
ntk_path = os.path.join(NTK_DIR, ntk_file)
with open(ntk_path) as f:
    ntk_data = json.load(f)

task_angle = TASK_ANGLE if TASK_ANGLE is not None else ntk_data["task_angle"]
print(f"Using eigendecomposition: {NTK_KEY}")
print(f"  step={eigs['snapshot_step']}  N={eigs['n_samples']}  task_angle={task_angle}°")

# ── Reconstruct residuals ─────────────────────────────────────────────────────

x   = torch.tensor(ntk_data["x"],   dtype=torch.float32)   # (N, 2)
f_x = torch.tensor(ntk_data["f_x"], dtype=torch.float32)   # (N,)

K = x.shape[1]
rad    = math.radians(task_angle)
w_task = torch.tensor([math.cos(rad), math.sin(rad)])
y      = x @ w_task     # (N,) — recompute for the requested task angle
r      = y - f_x        # (N,) residuals

# ── Retrieve eigenvectors ─────────────────────────────────────────────────────

if NTK_KEY not in eigs:
    available = [k for k in eigs.keys()
                 if k not in ("source_file","experiment","condition",
                               "seed_idx","snapshot_step","n_samples","task_angle")]
    raise KeyError(f"NTK_KEY '{NTK_KEY}' not found. Available: {available}")

layer_eigs = eigs[NTK_KEY]
eigenvalues  = torch.tensor(layer_eigs["eigenvalues"],  dtype=torch.float32)   # (N,)
eigenvectors = torch.tensor(layer_eigs["eigenvectors"], dtype=torch.float32)   # (N, N)

# ── Compute alignment ─────────────────────────────────────────────────────────

align = residual_alignment(r, eigenvectors)   # (N,)

# Learning speed proxy: λ_k * alignment_k
speed = eigenvalues * align

# ── Print summary ─────────────────────────────────────────────────────────────

print(f"\nResidual alignment  (task_angle={task_angle}°, NTK={NTK_KEY})")
print(f"  ||r|| = {r.norm().item():.6f}")
print(f"  {'k':>4}  {'λ_k':>10}  {'align_k':>10}  {'speed_k':>10}  {'cum_align':>10}")
cum = 0.0
for k in range(min(20, len(eigenvalues))):
    cum += align[k].item()
    print(f"  {k:>4}  {eigenvalues[k].item():>10.4f}  "
          f"{align[k].item():>10.4f}  {speed[k].item():>10.4f}  {cum:>10.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────

out = {
    "source_eigs":    EIGS_FILE,
    "ntk_key":        NTK_KEY,
    "task_angle":     task_angle,
    "snapshot_step":  eigs["snapshot_step"],
    "residual_norm":  float(r.norm().item()),
    "eigenvalues":    eigenvalues.tolist(),
    "alignments":     align.tolist(),
    "speed":          speed.tolist(),
    "cum_alignment":  torch.cumsum(align, 0).tolist(),
}

stem     = EIGS_FILE.replace("eigs_", "align_").replace(".json", f"_{NTK_KEY.replace('.','_')}_task{task_angle}.json")
out_path = os.path.join(NTK_DIR, stem)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}")
