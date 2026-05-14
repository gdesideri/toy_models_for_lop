"""
plot_ntk_over_time.py — Visualise NTK evolution over training (norm_growth_270).

Fig A — NTK trace over time: full, fc1, fc2  (one panel per condition)
Fig B — Top-k eigenvalue evolution over time  (one panel per condition)
Fig C — Eigenvalue spectrum at each 2k-step snapshot  (heatmap: step × eigenmode)
Fig D — Residual alignment over time: task A (0°) vs task B (270°)

Run with:  uv run python notebooks/plot_ntk_over_time.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
NTK_DIR = os.path.join(RESULTS, "ntk")
OUT_DIR = NTK_DIR

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3})

with open(os.path.join(NTK_DIR, "ntk_over_time.json")) as f:
    data = json.load(f)

CONDITIONS   = list(data.keys())
COND_COLORS  = {"ntest2000": "tab:blue", "ntest5000": "tab:orange", "ntest7000": "tab:green"}
PHASE_BG     = {"init": "white", "align": "#dce9f7",
                "norm_grow": "#fdf0d5", "test": "#fde8e8"}
TOP_K        = 6    # how many eigenvalues to show in Fig B


def phase_spans(snaps, ax):
    """Shade background by phase."""
    prev_phase, prev_step = None, None
    for s in snaps:
        if s["phase"] != prev_phase:
            if prev_phase is not None:
                ax.axvspan(prev_step, s["step"],
                           alpha=0.12, color={"align":"tab:blue",
                                              "norm_grow":"tab:orange",
                                              "test":"tab:red",
                                              "init":"white"}.get(prev_phase,"white"),
                           lw=0)
            prev_phase = s["phase"]
            prev_step  = s["step"]
    if prev_phase:
        ax.axvspan(prev_step, snaps[-1]["step"],
                   alpha=0.12, color={"align":"tab:blue","norm_grow":"tab:orange",
                                      "test":"tab:red","init":"white"}.get(prev_phase,"white"), lw=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure A — NTK trace over time
# ═══════════════════════════════════════════════════════════════════════════════

fig_a, axes = plt.subplots(1, len(CONDITIONS), figsize=(5*len(CONDITIONS), 4),
                            sharey=False)
if len(CONDITIONS) == 1:
    axes = [axes]

for ax, cond_key in zip(axes, CONDITIONS):
    snaps = data[cond_key]["snapshots"]
    steps = [s["step"] for s in snaps]

    trace_full = [s["trace_full"] for s in snaps]
    trace_fc1  = [s["layer_traces"].get("fc1.weight", 0) for s in snaps]
    trace_fc2  = [s["layer_traces"].get("fc2.weight", 0) for s in snaps]

    phase_spans(snaps, ax)
    ax.plot(steps, trace_full, color="black", lw=2,         label="Full NTK")
    ax.plot(steps, trace_fc1,  color="tab:orange", lw=1.5,  label="fc1 contrib")
    ax.plot(steps, trace_fc2,  color="tab:purple", lw=1.5,  label="fc2 contrib")

    ax.set_xlabel("Training step"); ax.set_ylabel("Trace  (= ∑ λ_k)")
    ax.set_title(f"{cond_key}  (n_test={data[cond_key]['n_test']})", fontsize=9)
    if ax == axes[0]:
        ax.legend(fontsize=8, frameon=False)

patches = [mpatches.Patch(color="tab:blue",   alpha=0.3, label="align"),
           mpatches.Patch(color="tab:orange", alpha=0.3, label="norm_grow"),
           mpatches.Patch(color="tab:red",    alpha=0.3, label="test")]
fig_a.legend(handles=patches, loc="upper right", fontsize=8, frameon=False)
plt.suptitle("Figure A — NTK trace over training (norm_growth_270)\n"
             "trace = ∑ eigenvalues  |  shading = training phase",
             fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, "figA_ntk_over_time.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure B — Top-k eigenvalue evolution
# ═══════════════════════════════════════════════════════════════════════════════

eig_colors = matplotlib.cm.plasma(np.linspace(0.1, 0.9, TOP_K))

fig_b, axes = plt.subplots(1, len(CONDITIONS), figsize=(5*len(CONDITIONS), 4))
if len(CONDITIONS) == 1:
    axes = [axes]

for ax, cond_key in zip(axes, CONDITIONS):
    snaps = data[cond_key]["snapshots"]
    steps = [s["step"] for s in snaps]
    eigs  = np.array([s["eigenvalues"] for s in snaps])   # (T, N)

    phase_spans(snaps, ax)
    for k in range(TOP_K):
        ax.plot(steps, eigs[:, k], color=eig_colors[k], lw=1.8,
                label=f"λ_{k}")

    ax.set_yscale("log")
    ax.set_xlabel("Training step"); ax.set_ylabel("Eigenvalue (log)")
    ax.set_title(f"{cond_key}", fontsize=9)
    if ax == axes[0]:
        ax.legend(fontsize=7, frameon=False, ncol=2)

plt.suptitle(f"Figure B — Top-{TOP_K} NTK eigenvalues over training\n"
             "shading = training phase  |  log scale",
             fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, "figB_ntk_over_time.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure C — Eigenvalue spectrum heatmap  (step × eigenmode)
# ═══════════════════════════════════════════════════════════════════════════════

N_SHOW = 16   # show top-N eigenmodes

fig_c, axes = plt.subplots(1, len(CONDITIONS), figsize=(5*len(CONDITIONS), 4))
if len(CONDITIONS) == 1:
    axes = [axes]

for ax, cond_key in zip(axes, CONDITIONS):
    snaps = data[cond_key]["snapshots"]
    steps = [s["step"] for s in snaps]
    eigs  = np.array([s["eigenvalues"][:N_SHOW] for s in snaps])   # (T, N_SHOW)

    # log scale for visibility
    eigs_log = np.log10(np.clip(eigs, 1e-6, None))

    im = ax.imshow(eigs_log.T, aspect="auto", origin="upper",
                   cmap="YlOrRd", interpolation="nearest",
                   extent=[steps[0], steps[-1], N_SHOW-0.5, -0.5])
    plt.colorbar(im, ax=ax, label="log₁₀(λ_k)", fraction=0.046)

    # phase boundary lines
    pb = data[cond_key]["phase_boundaries"]
    ax.axvline(pb["align_end"],     color="black", lw=1.5, ls="--", alpha=0.7)
    ax.axvline(pb["norm_grow_end"], color="black", lw=1.5, ls="--", alpha=0.7)

    ax.set_xlabel("Training step"); ax.set_ylabel("Eigenmode k")
    ax.set_title(f"{cond_key}", fontsize=9)
    ax.set_yticks(range(0, N_SHOW, 2))
    ax.grid(False)

plt.suptitle(f"Figure C — NTK eigenvalue spectrum over time  (top {N_SHOW} modes)\n"
             "colour = log₁₀(λ_k)  |  dashed lines = phase boundaries",
             fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, "figC_ntk_over_time.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure D — Residual alignment heatmap: step × eigenmode
#   3 rows per condition: alignment task A | alignment task B | speed task B
# ═══════════════════════════════════════════════════════════════════════════════

N_MODES = 12   # how many eigenmodes to show

# 3 rows: align_A, align_B, speed_B (= λ_k * align_B)
fig_d, axes = plt.subplots(3, len(CONDITIONS),
                            figsize=(5.5*len(CONDITIONS), 9), squeeze=False)

for col, cond_key in enumerate(CONDITIONS):
    snaps = data[cond_key]["snapshots"]
    steps = np.array([s["step"] for s in snaps])
    pb    = data[cond_key]["phase_boundaries"]

    align_A = np.array([s["align_taskA"][:N_MODES] for s in snaps])
    align_B = np.array([s["align_taskB"][:N_MODES] for s in snaps])
    eigvals = np.array([s["eigenvalues"][:N_MODES]  for s in snaps])
    speed_B = eigvals * align_B

    extent = [steps[0], steps[-1], N_MODES - 0.5, -0.5]

    for row, (mat, label, cmap, vmax_val) in enumerate([
        (align_A, "Alignment — Task A (0°)",   "Blues",  1.0),
        (align_B, "Alignment — Task B (270°)",  "Reds",   1.0),
        (np.log10(speed_B + 1e-6),
                  "log₁₀(λ_k · align_B)  [learning speed for Task B]",
                  "YlOrRd", None),
    ]):
        ax = axes[row][col]
        vmax = vmax_val if vmax_val is not None else mat.max()
        vmin = 0 if vmax_val is not None else mat.min()
        im = ax.imshow(mat.T, aspect="auto", origin="upper", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest",
                       extent=extent)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.axvline(pb["align_end"],     color="black", lw=1.5, ls="--", alpha=0.7)
        ax.axvline(pb["norm_grow_end"], color="black", lw=1.5, ls="--", alpha=0.7)
        ax.set_yticks(range(0, N_MODES, 2))
        ax.set_xlabel("Training step")
        ax.set_ylabel("Eigenmode k")
        ax.grid(False)

        title = f"{cond_key}" if row == 0 else ""
        ax.set_title(f"{title}\n{label}" if title else label, fontsize=8)

plt.suptitle(
    "Figure D — Residual alignment with NTK eigenmodes over training\n"
    "top: task A residual (0°)  |  mid: task B residual (270°)  |  "
    "bottom: learning speed = λ_k · align_B\n"
    "dashed lines = phase boundaries  |  bright = high alignment / fast learning",
    fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, "figD_ntk_over_time.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure E — Prefix vs Fresh: alignment and trace comparison
#   For each condition, at the START of phase 3 (test):
#     - bar chart: alignment_k for prefix vs fresh (averaged over fresh seeds)
#     - text: trace and residual norm comparison
# ═══════════════════════════════════════════════════════════════════════════════

fig_e, axes = plt.subplots(2, len(CONDITIONS),
                            figsize=(5.5*len(CONDITIONS), 7), squeeze=False)

bar_w = 0.35
xs    = np.arange(N_MODES)

for col, cond_key in enumerate(CONDITIONS):
    snaps = data[cond_key]["snapshots"]
    pb    = data[cond_key]["phase_boundaries"]
    test_start = pb["norm_grow_end"]

    # find prefix snapshot at start of test phase
    test_snaps = [s for s in snaps if s["step"] >= test_start]
    prefix_snap = test_snaps[0] if test_snaps else snaps[-1]

    prefix_align = np.array(prefix_snap["align_taskB"][:N_MODES])
    prefix_eigs  = np.array(prefix_snap["eigenvalues"][:N_MODES])
    prefix_trace = prefix_snap["trace_full"]
    prefix_rB    = prefix_snap["residual_norm_B"]

    fresh        = data[cond_key]["fresh_ntk"]
    fresh_align  = np.array(fresh["align_taskB_mean"][:N_MODES])
    fresh_eigs   = np.array(fresh["eigenvalues_mean"][:N_MODES])
    fresh_trace  = fresh["trace_mean"]
    fresh_rB     = fresh["residual_norm_B_mean"]

    # row 0 — alignment comparison
    ax0 = axes[0][col]
    ax0.bar(xs - bar_w/2, prefix_align, width=bar_w, color="tab:red",
            alpha=0.8, label="prefix")
    ax0.bar(xs + bar_w/2, fresh_align,  width=bar_w, color="tab:blue",
            alpha=0.8, label="fresh (mean)")
    ax0.set_xticks(xs); ax0.set_xlabel("Eigenmode k")
    ax0.set_ylabel("Alignment (v_k^T r_B)² / ||r_B||²")
    ax0.set_title(
        f"{cond_key}\n"
        f"prefix trace={prefix_trace:.0f}  fresh trace={fresh_trace:.0f}\n"
        f"prefix |r_B|={prefix_rB:.2f}  fresh |r_B|={fresh_rB:.2f}",
        fontsize=8)
    if col == 0:
        ax0.legend(fontsize=8, frameon=False)

    # row 1 — eigenvalue comparison
    ax1 = axes[1][col]
    ax1.bar(xs - bar_w/2, prefix_eigs, width=bar_w, color="tab:red",   alpha=0.8, label="prefix")
    ax1.bar(xs + bar_w/2, fresh_eigs,  width=bar_w, color="tab:blue",  alpha=0.8, label="fresh")
    ax1.set_yscale("log")
    ax1.set_xticks(xs); ax1.set_xlabel("Eigenmode k")
    ax1.set_ylabel("Eigenvalue λ_k  (log)")
    ax1.set_title("Eigenvalue spectrum at test-phase start", fontsize=8)
    if col == 0:
        ax1.legend(fontsize=8, frameon=False)

plt.suptitle(
    "Figure E — Prefix vs Fresh NTK at start of test phase (task B = 270°)\n"
    "top: residual alignment per eigenmode  |  bottom: eigenvalue spectrum\n"
    "Key: same alignment structure but prefix has much larger trace and ||r_B||",
    fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, "figE_ntk_over_time.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close()

print("\nAll figures saved.")
