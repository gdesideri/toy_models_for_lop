"""
compute_ntk_over_time.py — Compute NTK every 2k steps for norm_growth_270.

For each condition (ntest2000, ntest5000, ntest7000) and each seed:
  - Finds snapshots closest to multiples of STEP_INTERVAL
  - Computes full NTK + per-layer NTK at each snapshot
  - Computes eigendecomposition
  - Computes residual alignment for task A (0°) and task B (270°)
  - Saves compact summary (no full matrices — only eigenvalues, traces, alignment)

Saves to: results/ntk/ntk_over_time.json

Run with:  uv run python notebooks/compute_ntk_over_time.py
"""
import json, os, sys, math, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models import MinimalNet
from src.ntk    import (compute_ntk, compute_ntk_per_layer,
                         ntk_eigendecomposition, residual_alignment)

# ── Config ────────────────────────────────────────────────────────────────────

STEP_INTERVAL = 2_000      # compute NTK at 0, 2k, 4k, …
N_SAMPLES     = 64         # data points for NTK (smaller = faster)
SEED_IDX      = 0          # which seed to analyse
K_DIM         = 2

W0 = torch.tensor([ 1.0,  0.0])   # task A direction (align phase)
W3 = torch.tensor([ 0.0, -1.0])   # task B direction (test phase, 270°)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
NTK_DIR = os.path.join(RESULTS, "ntk")
os.makedirs(NTK_DIR, exist_ok=True)

# ── Fixed data batch (same for all snapshots) ─────────────────────────────────

torch.manual_seed(0)
X   = torch.randn(N_SAMPLES, K_DIM)
Y_A = X @ W0    # residuals for task A
Y_B = X @ W3    # residuals for task B

# ── Load experiment ───────────────────────────────────────────────────────────

exp_path = os.path.join(RESULTS, "norm_growth_270", "exp_norm_growth_270.json")
with open(exp_path) as f:
    exp = json.load(f)

conditions = list(exp["conditions"].keys())   # ["ntest2000", "ntest5000", "ntest7000"]
print(f"Conditions: {conditions}")
print(f"Seed: {SEED_IDX}  |  N_samples: {N_SAMPLES}  |  interval: {STEP_INTERVAL}\n")


# ── Helper: reconstruct model from snapshot ───────────────────────────────────

def load_model(snap: dict) -> MinimalNet:
    model = MinimalNet(d_in=K_DIM, hidden=4, bias=False)
    w1 = torch.tensor(snap["fc1_weight"], dtype=torch.float32)
    w2 = torch.tensor(snap["fc2_weight"], dtype=torch.float32)
    model.fc1.weight.data.copy_(w1)
    model.fc2.weight.data.copy_(w2.reshape(1, -1))
    model.eval()
    return model


# ── Helper: compute NTK summary at one snapshot ───────────────────────────────

def ntk_summary(model: MinimalNet, phase: str) -> dict:
    with torch.no_grad():
        f_x = model(X)     # (N,)

    # residuals for both tasks
    r_A = Y_A - f_x
    r_B = Y_B - f_x

    # full NTK
    K_full    = compute_ntk(model, X)            # (N, N)
    K_layers  = compute_ntk_per_layer(model, X)  # dict name→(N,N)

    # eigendecompose full NTK
    vals, vecs = ntk_eigendecomposition(K_full)

    # alignment
    align_A = residual_alignment(r_A, vecs)
    align_B = residual_alignment(r_B, vecs)

    # per-layer traces and top eigenvalue
    layer_traces = {name: float(K.trace().item()) for name, K in K_layers.items()}
    layer_top    = {}
    for name, K in K_layers.items():
        v, _ = ntk_eigendecomposition(K)
        layer_top[name] = float(v[0].item())

    return {
        "trace_full":    float(K_full.trace().item()),
        "layer_traces":  layer_traces,
        "layer_top_eig": layer_top,
        "eigenvalues":   vals.tolist(),         # (N,) descending
        "align_taskA":   align_A.tolist(),      # (N,) alignment with 0° residual
        "align_taskB":   align_B.tolist(),      # (N,) alignment with 270° residual
        "residual_norm_A": float(r_A.norm().item()),
        "residual_norm_B": float(r_B.norm().item()),
        "f_x_mean":      float(f_x.mean().item()),
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

all_results = {}

for cond_key in conditions:
    cond      = exp["conditions"][cond_key]
    seed_data = cond["seeds"][SEED_IDX]
    n_test    = cond["n_test"]

    # total steps for this condition
    total_steps = exp["config"]["n_align"] + exp["config"]["n_norm"] + n_test
    target_steps = list(range(0, total_steps + 1, STEP_INTERVAL))
    if total_steps not in target_steps:
        target_steps.append(total_steps)

    # index snapshots by step
    snaps = {s["step"]: s for s in seed_data["snapshots"]
             if s.get("fc1_weight") is not None and s.get("fc2_weight") is not None}

    phase_boundaries = {
        "align_end":     exp["config"]["n_align"],
        "norm_grow_end": exp["config"]["n_align"] + exp["config"]["n_norm"],
    }

    print(f"\n{'─'*60}")
    print(f"Condition: {cond_key}  (n_test={n_test}, total={total_steps} steps)")
    print(f"Target steps: {target_steps}")

    cond_results = {
        "n_test":   n_test,
        "config":   exp["config"],
        "phase_boundaries": phase_boundaries,
        "snapshots": [],
    }

    for tgt in target_steps:
        # find nearest available snapshot
        available = sorted(snaps.keys())
        nearest   = min(available, key=lambda s: abs(s - tgt))
        snap      = snaps[nearest]
        phase     = snap.get("phase", "?")

        t0    = time.time()
        model = load_model(snap)
        summary = ntk_summary(model, phase)
        dt    = time.time() - t0

        top5 = [f"{v:.2f}" for v in summary["eigenvalues"][:5]]
        print(f"  step={nearest:6d}  phase={phase:10s}  "
              f"trace={summary['trace_full']:8.2f}  "
              f"top-5={top5}  "
              f"|r_A|={summary['residual_norm_A']:.3f}  "
              f"|r_B|={summary['residual_norm_B']:.3f}  "
              f"({dt:.1f}s)")

        cond_results["snapshots"].append({
            "step":    nearest,
            "phase":   phase,
            **summary,
        })

    # ── Fresh model NTK at start of test phase ────────────────────────────────
    # Compare prefix alignment with a freshly-initialised model to explain LoP.
    # We average over N_FRESH_SEEDS random inits to get a stable estimate.

    N_FRESH_SEEDS = 20
    fresh_align_B_list  = []
    fresh_eigs_list     = []
    fresh_trace_list    = []
    fresh_res_norm_list = []

    for fs in range(N_FRESH_SEEDS):
        torch.manual_seed(1000 + fs)
        f_model = MinimalNet(d_in=K_DIM, hidden=4, bias=False)
        f_model.eval()
        with torch.no_grad():
            ff_x = f_model(X)
        r_B_fresh = Y_B - ff_x
        K_fresh = compute_ntk(f_model, X)
        vals_f, vecs_f = ntk_eigendecomposition(K_fresh)
        fresh_align_B_list.append(residual_alignment(r_B_fresh, vecs_f).tolist())
        fresh_eigs_list.append(vals_f.tolist())
        fresh_trace_list.append(float(K_fresh.trace().item()))
        fresh_res_norm_list.append(float(r_B_fresh.norm().item()))

    import numpy as _np
    fresh_align_B_mean = _np.mean(fresh_align_B_list, axis=0).tolist()
    fresh_eigs_mean    = _np.mean(fresh_eigs_list,    axis=0).tolist()

    cond_results["fresh_ntk"] = {
        "align_taskB_mean":  fresh_align_B_mean,
        "eigenvalues_mean":  fresh_eigs_mean,
        "trace_mean":        float(_np.mean(fresh_trace_list)),
        "residual_norm_B_mean": float(_np.mean(fresh_res_norm_list)),
        "n_seeds":           N_FRESH_SEEDS,
    }
    print(f"\n  [fresh NTK avg over {N_FRESH_SEEDS} seeds]"
          f"  trace={cond_results['fresh_ntk']['trace_mean']:.2f}"
          f"  |r_B|={cond_results['fresh_ntk']['residual_norm_B_mean']:.3f}"
          f"  top-4 align={[f'{x:.3f}' for x in fresh_align_B_mean[:4]]}")

    all_results[cond_key] = cond_results

# ── Save ───────────────────────────────────────────────────────────────────────

out_path = os.path.join(NTK_DIR, "ntk_over_time.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {out_path}")
