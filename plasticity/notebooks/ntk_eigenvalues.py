"""
ntk_eigenvalues.py — Eigendecompose a saved NTK matrix.

Loads a precomputed NTK JSON (from compute_ntk.py), computes the full
eigendecomposition of the full NTK and each per-layer NTK, and saves:
  - eigenvalues  (descending)
  - eigenvectors (columns = eigenvectors)
  - explained variance ratio
  - effective rank

Config
──────
Set NTK_FILE to the path of the NTK JSON produced by compute_ntk.py.

Run with:  uv run python notebooks/ntk_eigenvalues.py
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.ntk import ntk_eigendecomposition

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
NTK_DIR  = os.path.join(RESULTS, "ntk")

# Set this to the filename produced by compute_ntk.py (without path)
NTK_FILE = None   # e.g. "rotating_target_steps10000_cycles1_s0_step10000_N128_task0.json"
                  # if None → uses the most recent file in results/ntk/

# ── Load ───────────────────────────────────────────────────────────────────────

if NTK_FILE is None:
    files = sorted([f for f in os.listdir(NTK_DIR) if f.endswith(".json")
                    and not f.startswith("eigs_")])
    if not files:
        raise FileNotFoundError(f"No NTK JSON files found in {NTK_DIR}")
    NTK_FILE = files[-1]
    print(f"Auto-selected: {NTK_FILE}")

ntk_path = os.path.join(NTK_DIR, NTK_FILE)
with open(ntk_path) as f:
    data = json.load(f)

print(f"Loaded NTK: {NTK_FILE}")
print(f"  experiment={data['experiment']}  condition={data['condition']}"
      f"  step={data['snapshot_step']}  N={data['n_samples']}")

# ── Eigendecompose ────────────────────────────────────────────────────────────

def decompose_and_summarise(K_list: list, label: str) -> dict:
    K    = torch.tensor(K_list, dtype=torch.float64)
    vals, vecs = ntk_eigendecomposition(K.float())
    vals_np = vals.numpy()
    vecs_np = vecs.numpy()

    total_var   = vals_np.sum()
    cum_var     = np.cumsum(vals_np) / (total_var + 1e-12)
    eff_rank    = int((vals_np > 1e-4 * vals_np[0]).sum())
    entropy_rank = float(np.exp(
        -np.sum((vals_np / (total_var + 1e-12)) *
                 np.log(vals_np / (total_var + 1e-12) + 1e-30))
    ))

    print(f"\n  [{label}]")
    print(f"    trace={total_var:.4f}  eff_rank={eff_rank}"
          f"  entropy_rank={entropy_rank:.2f}")
    print(f"    top eigenvalues: {vals_np[:8].round(4)}")

    return {
        "label":         label,
        "eigenvalues":   vals_np.tolist(),
        "eigenvectors":  vecs_np.tolist(),
        "trace":         float(total_var),
        "eff_rank":      eff_rank,
        "entropy_rank":  float(entropy_rank),
        "cum_var_ratio": cum_var.tolist(),
    }

print("\nEigendecomposing...")

results = {
    "source_file": NTK_FILE,
    "experiment":  data["experiment"],
    "condition":   data["condition"],
    "seed_idx":    data["seed_idx"],
    "snapshot_step": data["snapshot_step"],
    "n_samples":   data["n_samples"],
    "task_angle":  data["task_angle"],
    "full":        decompose_and_summarise(data["K_full"], "full NTK"),
}

for layer_name, K_layer in data["K_layers"].items():
    results[layer_name] = decompose_and_summarise(K_layer, layer_name)

# ── Save ───────────────────────────────────────────────────────────────────────

out_name = "eigs_" + NTK_FILE
out_path = os.path.join(NTK_DIR, out_name)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_path}")
