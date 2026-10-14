"""
plot_norm_growth_270.py — Visualise exp_norm_growth_270.json

For each n_test condition generates:

  figA_norm_growth_270_ntest<N>.png  — fc1 directions at 8 snapshots, seed 0
  figB_norm_growth_270_ntest<N>.png  — loss curves, seed 0
  figC_norm_growth_270_ntest<N>.png  — norms + signed atan2 angle over time
  figE_norm_growth_270_ntest<N>.png  — effective contribution vectors
                                       (fc2[u]*fc1[u]) during phase 3

One summary figure:
  figD_norm_growth_270.png           — LoP gap vs n_test, compared with 180° case

Run with:  uv run python visualization_scripts/toy_models/plot_norm_growth_270.py
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
OUT_DIR = os.path.join(RESULTS, "norm_growth_270")
os.makedirs(OUT_DIR, exist_ok=True)

def load(name, subdir=None):
    base = os.path.join(RESULTS, subdir) if subdir else RESULTS
    with open(os.path.join(base, f"{name}.json")) as f:
        return json.load(f)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3})

data = load("norm_growth_270/exp_norm_growth_270")
cfg  = data["config"]
SEED_IDX = 0

W0 = np.array([ 1.0,  0.0])   # 0°
W3 = np.array([ 0.0, -1.0])   # 270° — target
UNIT_COLORS = ["tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
SEED_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
PHASE_BG    = {"init": "white", "align": "#dce9f7",
               "norm_grow": "#fdf0d5", "test": "#fde8e8"}


def parse_snaps(seed_data):
    valid = [(s["step"], s["phase"],
              np.array(s["fc1_weight"]),
              np.array(s["fc2_weight"]),
              s["fc1_norms"],
              s["fc1_dir_angles"],
              s.get("eff_vecs"))
             for s in seed_data["snapshots"] if s.get("fc1_weight") is not None]
    steps     = np.array([v[0] for v in valid])
    phases    = [v[1] for v in valid]
    fc1s      = [v[2] for v in valid]
    fc2s      = [v[3] for v in valid]
    norms     = [v[4] for v in valid]
    dir_angs  = [v[5] for v in valid]
    eff_vecs  = [v[6] for v in valid]
    p1_end = max((s for s, p in zip(steps, phases) if p == "align"),      default=0)
    p2_end = max((s for s, p in zip(steps, phases) if p == "norm_grow"),  default=0)
    return steps, phases, fc1s, fc2s, norms, dir_angs, eff_vecs, p1_end, p2_end


# ── per-condition figures ──────────────────────────────────────────────────────

for cond_key, cond in data["conditions"].items():
    n_test    = cond["n_test"]
    seed_data = cond["seeds"][SEED_IDX]
    tag       = cond_key
    n_units   = cfg["hidden"]

    steps, phases, fc1s, fc2s, norms_t, dir_angs, eff_vecs, p1_end, p2_end = \
        parse_snaps(seed_data)

    p1 = seed_data["phase1_curve"]
    p2 = seed_data["phase2_curve"]
    pc = seed_data["prefix_curve"]
    fc = seed_data["fresh_curve"]
    lop = seed_data["lop_gap"]

    # ── Figure A — fc1 directions ──────────────────────────────────────────

    n_cols = 8
    idxs   = np.linspace(0, len(steps)-1, n_cols, dtype=int)
    cmap   = matplotlib.cm.RdYlBu_r

    fig_a, axes_a = plt.subplots(1, n_cols, figsize=(2.4*n_cols, 3.2))
    for col, si in enumerate(idxs):
        ax    = axes_a[col]
        phase = phases[si]
        fc1   = fc1s[si]
        step  = steps[si]

        ax.set_facecolor(PHASE_BG.get(phase, "white"))
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)

        target = W3 if phase == "test" else W0
        ax.annotate("", xy=target*1.5, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

        for ui in range(n_units):
            vec  = fc1[ui]
            norm = np.linalg.norm(vec) + 1e-10
            uvec = vec / norm
            ax.annotate("", xy=uvec*min(norm, 4.5), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=UNIT_COLORS[ui], lw=1.5))

        pname = {"init":"P0","align":"P1","norm_grow":"P2","test":"P3"}.get(phase, phase)
        ax.set_title(f"{pname} step {step}\nnorms:{[f'{norms_t[si][u]:.1f}' for u in range(n_units)]}",
                     fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=UNIT_COLORS[u], label=f'unit {u}')
               for u in range(n_units)]
    fig_a.legend(handles=handles, loc='lower center',
                 ncol=n_units, fontsize=8, frameon=False,
                 bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f"Figure A — fc1 directions  [270° target, n_test={n_test}, seed {SEED_IDX}]\n"
                 f"P1=align  P2=norm×{cfg['scale']}  P3=270° target w3=(0,-1)  |  axes ±5",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"figA_norm_growth_270_{tag}.png")
    fig_a.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close(fig_a)

    # ── Figure B — loss curves ─────────────────────────────────────────────

    p2_rsc = [(s, v / cfg["scale"]**2) for s, v in p2]

    fig_b, ax = plt.subplots(figsize=(12, 4))
    ax.plot([s for s,v in p1], [v for s,v in p1],
            color="tab:blue", lw=2, label="prefix — phase 1  y=w0·x")
    ax.plot([s for s,v in p2_rsc], [v for s,v in p2_rsc],
            color="tab:orange", lw=2,
            label=f"prefix — phase 2  y={cfg['scale']}·w0·x  (÷{cfg['scale']}²)")
    ax.plot([s for s,v in pc], [v for s,v in pc],
            color="tab:red", lw=2, label="prefix — phase 3  y=w3·x  (270°)")
    ax.plot([s for s,v in fc], [v for s,v in fc],
            color="tab:red", lw=2, ls="--", alpha=0.5, label="fresh  — phase 3")
    ax.axvline(p1_end, color="gray", lw=1, ls="--", alpha=0.6)
    ax.axvline(p2_end, color="gray", lw=1, ls="--", alpha=0.6)
    ax.axvspan(0,      p1_end, alpha=0.06, color="tab:blue")
    ax.axvspan(p1_end, p2_end, alpha=0.06, color="tab:orange")
    ax.axvspan(p2_end, steps[-1]+500, alpha=0.06, color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("Global step"); ax.set_ylabel("MSE (log scale)")
    ax.legend(fontsize=8, frameon=False)
    plt.suptitle(f"Figure B — Loss curves  [270° target, n_test={n_test}, seed {SEED_IDX}]  "
                 f"LoP={lop:+.4f}",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"figB_norm_growth_270_{tag}.png")
    fig_b.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close(fig_b)

    # ── Figure C — norms + signed atan2 angle ─────────────────────────────

    fig_c, (ax_n, ax_a) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ui in range(n_units):
        ax_n.plot(steps, [norms_t[i][ui] for i in range(len(steps))],
                  color=UNIT_COLORS[ui], lw=1.5, label=f"unit {ui}")
        ax_a.plot(steps, [dir_angs[i][ui] for i in range(len(steps))],
                  color=UNIT_COLORS[ui], lw=1.5, label=f"unit {ui}")

    for ax in (ax_n, ax_a):
        ax.axvline(p1_end, color="gray", lw=1, ls="--", alpha=0.6)
        ax.axvline(p2_end, color="gray", lw=1, ls="--", alpha=0.6)
        ax.axvspan(0,      p1_end, alpha=0.05, color="tab:blue")
        ax.axvspan(p1_end, p2_end, alpha=0.05, color="tab:orange")
        ax.axvspan(p2_end, steps[-1]+200, alpha=0.05, color="tab:red")

    ax_n.set_ylabel("||w_i||  (fc1 unit norm)")
    ax_n.legend(fontsize=8, frameon=False, ncol=4)

    ax_a.axhline(   0, color="green",   lw=1.2, ls="--", alpha=0.7, label="0°   ← w0=(1,0)")
    ax_a.axhline( 180, color="crimson", lw=1.2, ls="--", alpha=0.7, label="±180° ← w2=(-1,0)")
    ax_a.axhline(-180, color="crimson", lw=1.2, ls="--", alpha=0.7)
    ax_a.axhline( -90, color="darkorange", lw=1.8, ls="-",  alpha=0.9,
                  label="-90°  ← w3=(0,-1)  [TARGET]")
    ax_a.set_ylim(-200, 200)
    ax_a.set_yticks([-180, -90, 0, 90, 180])
    ax_a.set_ylabel("Direction angle atan2(w_y, w_x)  (°)")
    ax_a.set_xlabel("Global step")
    ax_a.legend(fontsize=8, frameon=False, ncol=3)
    ax_n.set_title(f"P1: align  |  P2: norm×{cfg['scale']}  |  P3: 270° w3=(0,-1) ({n_test} steps)  "
                   "← rotation visible as drift toward -90°",
                   fontsize=8)

    plt.suptitle(f"Figure C — fc1 norms & signed direction angle  "
                 f"[270° target, n_test={n_test}, seed {SEED_IDX}]  LoP={lop:+.4f}\n"
                 "Units ROTATING: angles drift from 0°/±180° toward -90° in phase 3",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"figC_norm_growth_270_{tag}.png")
    fig_c.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close(fig_c)

    # ── Figure E — effective contribution vectors fc2*fc1 in phase 3 ──────
    # Shows the signed effective weight direction of each unit:
    #   eff_vec[u] = fc2[u] * fc1[u]   (K-dim vector)
    # For the w3=(0,-1) task we want eff_vec to point toward (0,-1).
    # Plotting eff_vec[u] as a 2D arrow in the (e1,e2) plane.

    p3_mask  = [i for i, ph in enumerate(phases) if ph == "test"]
    if not p3_mask:
        continue

    steps_p3 = steps[p3_mask]
    ev_p3    = [eff_vecs[i] for i in p3_mask if eff_vecs[i] is not None]
    if not ev_p3:
        continue

    n_snap = 6
    snap_idxs = np.linspace(0, len(ev_p3)-1, n_snap, dtype=int)

    fig_e, axes_e = plt.subplots(1, n_snap, figsize=(2.6*n_snap, 3.2))
    for col, si in enumerate(snap_idxs):
        ax   = axes_e[col]
        ev   = np.array(ev_p3[si])   # (HIDDEN, K)
        step = steps_p3[si + (len(p3_mask) - len(ev_p3))] if len(ev_p3) < len(p3_mask) \
               else steps_p3[si]

        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_facecolor("#fde8e8")

        # target direction
        ax.annotate("", xy=W3*1.5, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

        for ui in range(n_units):
            vec  = ev[ui]
            norm = np.linalg.norm(vec) + 1e-10
            uvec = vec / norm
            dot  = float(np.clip(abs(uvec @ W3), 0, 1))
            ang  = math.degrees(math.acos(dot)) / 90.0
            ax.annotate("", xy=uvec*min(norm, 4.5), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=cmap(ang), lw=1.5))

        step_in_p3 = step - p2_end
        ax.set_title(f"P3 +{step_in_p3} steps\nstep {step}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 90))
    sm2.set_array([])
    fig_e.colorbar(sm2, ax=axes_e.ravel().tolist(),
                   label="angle of fc2·fc1 to w3 target (°)",
                   fraction=0.015, pad=0.02)
    plt.suptitle(
        f"Figure E — Effective contribution  fc2[u]·fc1[u]  during phase 3\n"
        f"[270° target, n_test={n_test}, seed {SEED_IDX}]  LoP={lop:+.4f}\n"
        "Arrow = signed effective direction of each unit  |  "
        "black = w3=(0,-1) target  |  colour = angle to target",
        fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"figE_norm_growth_270_{tag}.png")
    fig_e.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close(fig_e)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure D — LoP comparison: 270° vs 180° (unscaled) vs 180° scaled
# ═══════════════════════════════════════════════════════════════════════════════

cond_keys   = list(data["conditions"].keys())
n_test_vals = [data["conditions"][k]["n_test"] for k in cond_keys]
n_seeds     = len(data["conditions"][cond_keys[0]]["seeds"])

try:
    data_180  = load("exp_norm_growth",        subdir="norm_growth")
    data_180s = load("exp_norm_growth_scaled",  subdir="norm_growth_scaled")
    has_180   = True
except FileNotFoundError:
    has_180   = False

fig_d, ax = plt.subplots(figsize=(10, 5))
bar_w = 0.22
x_pos = np.arange(len(cond_keys))

for si in range(n_seeds):
    lops = [data["conditions"][k]["seeds"][si]["lop_gap"] for k in cond_keys]
    ax.bar(x_pos + (si-1)*bar_w, lops, width=bar_w,
           color=SEED_COLORS[si], alpha=0.85, label=f"270° seed {si}")

means_270 = [data["conditions"][k]["mean_lop"] for k in cond_keys]
ax.plot(x_pos, means_270, "ko--", lw=1.5, ms=5, label="270° mean")

if has_180:
    means_180  = [data_180["conditions"][k]["mean_lop"]  for k in cond_keys]
    means_180s = [data_180s["conditions"][k]["mean_lop"] for k in cond_keys]
    ax.plot(x_pos, means_180,  "rs--", lw=1.5, ms=5, alpha=0.7,
            label="180° unscaled mean")
    ax.plot(x_pos, means_180s, "b^--", lw=1.5, ms=5, alpha=0.7,
            label="180° scaled mean")

ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{n//1000}k steps\n(phase 3)" for n in n_test_vals], fontsize=10)
ax.set_ylabel("LoP gap")
ax.legend(fontsize=9, frameon=False)

for x, m in zip(x_pos, means_270):
    ax.text(x, m + 0.3, f"{m:+.2f}", ha="center", fontsize=9, fontweight="bold")

plt.suptitle(
    "Figure D — LoP gap: 270° target vs 180° (unscaled and scaled)\n"
    f"phase1={cfg['n_align']} align  |  phase2={cfg['n_norm']} norm×{cfg['scale']}  |  "
    "270° has no dead zone → units actually rotate",
    fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, "figD_norm_growth_270.png")
fig_d.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}"); plt.close(fig_d)

print("\nAll figures saved.")
