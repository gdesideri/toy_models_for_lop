"""
plot_ntk.py — Visualise NTK matrix, eigenspectrum and residual alignment.

Loads the most recent NTK, eigendecomposition and alignment files from
results/ntk/ and produces:

  Fig A — NTK heatmap: full K and per-layer K_fc1, K_fc2  (side by side)
  Fig B — Eigenvalue spectrum: bar chart, log scale, per NTK
  Fig C — Cumulative explained variance by top-k eigenmodes
  Fig D — Residual alignment: alignment_k and speed_k = λ_k * align_k per mode

Run with:  uv run python notebooks/plot_ntk.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
NTK_DIR = os.path.join(RESULTS, "ntk")
OUT_DIR = NTK_DIR

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3})

# ── Auto-load most recent files ───────────────────────────────────────────────

def latest(prefix):
    files = sorted([f for f in os.listdir(NTK_DIR)
                    if f.startswith(prefix) and f.endswith(".json")])
    if not files:
        raise FileNotFoundError(f"No {prefix}*.json in {NTK_DIR}")
    return files[-1]

ntk_file   = latest("")       # plain NTK file (not eigs_ or align_)
# exclude eigs_ and align_ files
ntk_file   = sorted([f for f in os.listdir(NTK_DIR)
                      if f.endswith(".json")
                      and not f.startswith("eigs_")
                      and not f.startswith("align_")])[-1]
eigs_file  = latest("eigs_")
align_file = latest("align_")

with open(os.path.join(NTK_DIR, ntk_file))   as f: ntk   = json.load(f)
with open(os.path.join(NTK_DIR, eigs_file))  as f: eigs  = json.load(f)
with open(os.path.join(NTK_DIR, align_file)) as f: align = json.load(f)

print(f"NTK:   {ntk_file}")
print(f"Eigs:  {eigs_file}")
print(f"Align: {align_file}")

tag = (f"{ntk['experiment']}_{ntk['condition']}_s{ntk['seed_idx']}"
       f"_step{ntk['snapshot_step']}_N{ntk['n_samples']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure A — NTK heatmaps (full + per layer)
# ═══════════════════════════════════════════════════════════════════════════════

K_full   = np.array(ntk["K_full"])
K_layers = {name: np.array(K) for name, K in ntk["K_layers"].items()}
all_Ks   = {"Full NTK": K_full, **{f"K_{n}": v for n, v in K_layers.items()}}

n_plots = len(all_Ks)
fig_a, axes = plt.subplots(1, n_plots, figsize=(4.5*n_plots, 4.5))
if n_plots == 1:
    axes = [axes]

for ax, (label, K) in zip(axes, all_Ks.items()):
    vmax = np.abs(K).max()
    im   = ax.imshow(K, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{label}\ntrace={K.trace():.3f}", fontsize=9)
    ax.set_xlabel("sample j"); ax.set_ylabel("sample i")
    ax.grid(False)

plt.suptitle(
    f"Figure A — NTK heatmaps\n"
    f"{ntk['experiment']}  {ntk['condition']}  step={ntk['snapshot_step']}"
    f"  N={ntk['n_samples']}  task={ntk['task_angle']}°",
    fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, f"figA_ntk_{tag}.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure B — Eigenvalue spectrum
# ═══════════════════════════════════════════════════════════════════════════════

ntk_keys = ["full"] + [k for k in eigs
                        if k not in ("source_file","experiment","condition",
                                      "seed_idx","snapshot_step","n_samples",
                                      "task_angle","full")]

LAYER_COLORS = {"full": "tab:blue", "fc1.weight": "tab:orange", "fc2.weight": "tab:green"}

fig_b, ax = plt.subplots(figsize=(10, 4))
width = 0.8 / len(ntk_keys)
N    = eigs["full"]["eff_rank"] + 4   # show a bit beyond eff rank

for i, key in enumerate(ntk_keys):
    vals = np.array(eigs[key]["eigenvalues"])[:N]
    xs   = np.arange(len(vals)) + (i - len(ntk_keys)/2 + 0.5) * width
    col  = LAYER_COLORS.get(key, "tab:gray")
    ax.bar(xs, vals, width=width, color=col, alpha=0.8,
           label=f"{key}  (rank={eigs[key]['eff_rank']})")

ax.set_yscale("log")
ax.set_xlabel("Eigenvalue index k  (descending)")
ax.set_ylabel("λ_k  (log scale)")
ax.legend(fontsize=8, frameon=False)

plt.suptitle(
    f"Figure B — NTK eigenvalue spectrum\n"
    f"step={eigs['snapshot_step']}  N={eigs['n_samples']}",
    fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, f"figB_ntk_{tag}.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure C — Cumulative explained variance
# ═══════════════════════════════════════════════════════════════════════════════

fig_c, ax = plt.subplots(figsize=(8, 4))

for key in ntk_keys:
    cum  = np.array(eigs[key]["cum_var_ratio"])
    col  = LAYER_COLORS.get(key, "tab:gray")
    ax.plot(np.arange(1, len(cum)+1), cum, color=col, lw=2, label=key)

ax.axhline(0.95, color="gray", lw=1, ls="--", alpha=0.6, label="95% threshold")
ax.set_xlabel("Number of eigenmodes k")
ax.set_ylabel("Cumulative explained variance")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, frameon=False)

plt.suptitle("Figure C — Cumulative NTK variance by top-k eigenmodes",
             fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, f"figC_ntk_{tag}.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure D — Residual alignment and learning speed
# ═══════════════════════════════════════════════════════════════════════════════

eigenvalues = np.array(align["eigenvalues"])
alignments  = np.array(align["alignments"])
speed       = np.array(align["speed"])
cum_align   = np.array(align["cum_alignment"])

# Show top-K meaningful modes
K_show = min(30, int((eigenvalues > 1e-6 * eigenvalues[0]).sum()) + 4)
xs = np.arange(K_show)

fig_d, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# alignment
ax1.bar(xs, alignments[:K_show], color="tab:blue", alpha=0.8)
ax1.set_ylabel("Alignment  (v_k^T r)² / ||r||²")
ax1.set_title(f"task_angle={align['task_angle']}°  ||r||={align['residual_norm']:.4f}  "
              f"NTK={align['ntk_key']}")

# speed = λ_k * alignment_k
ax2.bar(xs, speed[:K_show], color="tab:orange", alpha=0.8)
ax2.set_ylabel("Speed  λ_k · alignment_k")

# cumulative alignment
ax3.plot(xs, cum_align[:K_show], color="tab:green", lw=2, marker="o", ms=4)
ax3.axhline(0.9, color="gray", lw=1, ls="--", alpha=0.5, label="90%")
ax3.set_ylabel("Cumulative alignment")
ax3.set_xlabel("Eigenmode k  (descending eigenvalue)")
ax3.legend(fontsize=8, frameon=False)

plt.suptitle(
    f"Figure D — Residual alignment with NTK eigenmodes\n"
    f"step={align['snapshot_step']}  |  "
    "high speed_k = fast learning of this residual component",
    fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, f"figD_ntk_{tag}.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close()

print("\nAll figures saved.")
