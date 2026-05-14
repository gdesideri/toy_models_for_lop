"""
plot_rotating_target.py — Visualise exp_rotating_target.json

For every experimental condition (each entry in steps_sweep and cycles_sweep)
a separate set of figures is produced:

  figA_<key>.png  — fc1 feature directions (arrows in 2D input space), seed 0
  figB_<key>.png  — prefix vs fresh loss curves per task, seed 0
  figD_<key>.png  — angle of each hidden unit to current task's w, seed 0

One summary figure is produced across all conditions:

  figC_lop.png    — LoP gap: steps sweep (mean±std) + per-seed bars at steps=2000

Run with:  uv run python notebooks/plot_rotating_target.py
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

def load(name):
    with open(os.path.join(RESULTS, f"{name}.json")) as f:
        return json.load(f)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3})

data = load("rotating_target/exp_rotating_target")

TASK_ANGLES  = [0, 90, 180, 270]
TASK_WS_NP   = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                          for a in TASK_ANGLES])
TASK_COLORS  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
UNIT_COLORS  = ["tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
SEED_IDX     = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_snaps(seed_data):
    snaps = seed_data["snapshots"]
    steps, task_ids, fc1s = [], [], []
    for s in snaps:
        w = s.get("fc1_weight")
        if w is None: continue
        arr = np.array(w)           # (H, 2)
        if arr.ndim != 2: continue
        steps.append(s["step"])
        task_ids.append(s.get("task_idx", -1))
        fc1s.append(arr)
    return np.array(steps), task_ids, fc1s

def angle_to_w(unit_vec, w_vec):
    """Angle in degrees between unit (2D) and w (2D)."""
    u = unit_vec / (np.linalg.norm(unit_vec) + 1e-10)
    dot = np.clip(u @ w_vec, -1, 1)
    return math.degrees(math.acos(abs(dot)))   # 0°=aligned, 90°=orthogonal

def lop_series(seed_results):
    steps = [b["lop_step"] for b in seed_results[0]["block_logs"]]
    gaps  = np.array([[b["lop_gap"] for b in r["block_logs"]] for r in seed_results])
    return np.array(steps), gaps

def n_tasks_from(cond):
    return len(cond["seeds"][0]["block_logs"])


# ── collect all conditions in order ───────────────────────────────────────────

ALL_CONDITIONS = []   # list of (sweep_name, key, cond_dict)
for key, cond in data["steps_sweep"].items():
    ALL_CONDITIONS.append(("steps_sweep", key, cond))

print(f"Conditions to plot: {[k for _,k,_ in ALL_CONDITIONS]}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure A — Feature directions per condition, seed 0
# ═══════════════════════════════════════════════════════════════════════════════

def plot_figA(key, cond, sweep_name):
    seed_data = cond["seeds"][SEED_IDX]
    steps_a, task_ids_a, fc1s_a = parse_snaps(seed_data)
    if not fc1s_a:
        print(f"  [figA] no snapshots for {key}, skipping"); return
    n_units = fc1s_a[0].shape[0]

    n_cols = 8
    idxs   = np.linspace(0, len(steps_a)-1, n_cols, dtype=int)
    cmap   = matplotlib.cm.RdYlBu_r

    fig, axes = plt.subplots(1, n_cols, figsize=(2.2*n_cols, 3))
    for col, si in enumerate(idxs):
        ax = axes[col]
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)

        ti   = task_ids_a[si]
        step = steps_a[si]
        w_task = TASK_WS_NP[ti] if 0 <= ti < 4 else TASK_WS_NP[0]

        ax.annotate("", xy=w_task, xytext=(0,0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

        fc1 = fc1s_a[si]
        for ui in range(n_units):
            vec  = fc1[ui]
            norm = np.linalg.norm(vec) + 1e-10
            uvec = vec / norm
            scale = min(norm, 1.4)
            ax.annotate("", xy=uvec*scale, xytext=(0,0),
                        arrowprops=dict(arrowstyle="-|>", color=UNIT_COLORS[ui], lw=1.5))

        ax.set_title(f"step {step}\ntask {ti} ({TASK_ANGLES[ti] if 0<=ti<4 else '?'}°)",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=UNIT_COLORS[u], label=f"unit {u}") for u in range(n_units)]
    fig.legend(handles=handles, loc="lower center", ncol=n_units,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    steps_per_task = cond["steps_per_task"]
    n_cycles       = cond["n_cycles"]
    plt.suptitle(
        f"Figure A — fc1 feature directions  [{key}]  seed {SEED_IDX}\n"
        f"black arrow = target w,  each colour = one hidden unit",
        fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS, "rotating_target", f"figA_{key}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure B — Loss curves per condition, seed 0
# ═══════════════════════════════════════════════════════════════════════════════

def plot_figB(key, cond, sweep_name):
    seed_data  = cond["seeds"][SEED_IDX]
    blocks     = seed_data["block_logs"]
    n_cycles   = cond["n_cycles"]
    n_tasks    = 4   # always 4 task directions

    fig, axes = plt.subplots(n_cycles, n_tasks,
                             figsize=(3.5*n_tasks, 3.0*n_cycles),
                             squeeze=False)

    for blk in blocks:
        row = blk["cycle"]
        col = blk["task_idx"]
        ax  = axes[row][col]
        ti  = blk["task_idx"]
        ps  = [s - blk["prefix_loss"][0][0] for s,_ in blk["prefix_loss"]]
        pv  = [v for _,v in blk["prefix_loss"]]
        fv  = [v for _,v in blk["fresh_loss"]]
        c   = TASK_COLORS[ti % len(TASK_COLORS)]
        ax.plot(ps, pv, color=c, lw=2, label="prefix")
        ax.plot(ps, fv, color=c, lw=2, ls="--", alpha=0.55, label="fresh")
        ax.set_yscale("log")
        ax.set_title(f"cy{row}  task{ti}  {TASK_ANGLES[ti]}°\nLoP={blk['lop_gap']:+.5f}",
                     fontsize=8)
        ax.set_xlabel("Step within task", fontsize=7)
        ax.set_ylabel("MSE", fontsize=7)
        if row == 0 and col == 0:
            ax.legend(fontsize=7, frameon=False)

    plt.suptitle(
        f"Figure B — MSE: prefix vs fresh  [{key}]  seed {SEED_IDX}\n"
        f"steps={cond['steps_per_task']}/task, n_cycles={n_cycles}, minimal 4-unit net",
        fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS, "rotating_target", f"figB_{key}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure D — Angle of hidden units per condition, seed 0
# ═══════════════════════════════════════════════════════════════════════════════

def plot_figD(key, cond, sweep_name):
    seed_data = cond["seeds"][SEED_IDX]
    steps_d, task_ids_d, fc1s_d = parse_snaps(seed_data)
    if not fc1s_d:
        print(f"  [figD] no snapshots for {key}, skipping"); return
    blocks_d = seed_data["block_logs"]
    n_units  = fc1s_d[0].shape[0]

    fig, ax = plt.subplots(figsize=(14, 4))

    for ui in range(n_units):
        angles = []
        for ti, fc1 in zip(task_ids_d, fc1s_d):
            w_t = TASK_WS_NP[ti] if 0 <= ti < 4 else TASK_WS_NP[0]
            angles.append(angle_to_w(fc1[ui], w_t))
        ax.plot(steps_d, angles, color=UNIT_COLORS[ui], lw=1.5, label=f"unit {ui}")

    # shade by task
    for blk in blocks_d:
        c = blk["prefix_loss"]
        if not c: continue
        ax.axvspan(c[0][0], c[-1][0], alpha=0.07,
                   color=TASK_COLORS[blk["task_idx"] % len(TASK_COLORS)])

    # cycle boundaries
    seen_cycles = set()
    for blk in blocks_d:
        cyc = blk["cycle"]
        if cyc > 0 and cyc not in seen_cycles and blk["task_idx"] == 0:
            if blk["prefix_loss"]:
                ax.axvline(blk["prefix_loss"][0][0], color="black", lw=1.2, alpha=0.5)
            seen_cycles.add(cyc)

    ax.axhline(0,  color="green",  lw=0.8, ls="--", alpha=0.6)
    ax.axhline(90, color="crimson",lw=0.8, ls="--", alpha=0.4)
    ax.set_ylim(-2, 95)
    ax.set_xlabel("Global step"); ax.set_ylabel("Angle to current task's w (°)")

    task_patches = [mpatches.Patch(color=TASK_COLORS[i], alpha=0.4,
                    label=f"Task {i}  {TASK_ANGLES[i]}°") for i in range(4)]
    ax.legend(handles=
        [mlines.Line2D([],[],color=UNIT_COLORS[u],lw=2,label=f"unit {u}") for u in range(n_units)] +
        [mlines.Line2D([],[],color="green",  lw=1,ls="--",label="0° aligned"),
         mlines.Line2D([],[],color="crimson",lw=1,ls="--",label="90° orthogonal")] +
        task_patches,
        fontsize=8, frameon=False, ncol=5)

    plt.suptitle(
        f"Figure D — Angle of hidden units to current task's w  [{key}]  seed {SEED_IDX}\n"
        f"0°=aligned, 90°=orthogonal  |  shaded = task region  |  vertical lines = cycle boundaries",
        fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS, "rotating_target", f"figD_{key}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close(fig)


# ── generate per-condition figures ────────────────────────────────────────────

for sweep_name, key, cond in ALL_CONDITIONS:
    print(f"\n[{sweep_name}]  {key}")
    plot_figA(key, cond, sweep_name)
    plot_figB(key, cond, sweep_name)
    plot_figD(key, cond, sweep_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure C — Summary: LoP gap across conditions  (single figure)
# ═══════════════════════════════════════════════════════════════════════════════

fig_c, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(12, 4))

colors_s  = ["tab:blue", "tab:orange", "tab:green"]
step_keys = ["steps2000_cycles1", "steps10000_cycles1", "steps50000_cycles1"]

# Panel 1: steps sweep mean±std
for col, key in zip(colors_s, step_keys):
    cond = data["steps_sweep"].get(key)
    if cond is None: continue
    _, gaps = lop_series(cond["seeds"])
    mean = gaps.mean(0); std = gaps.std(0)
    n_pts = len(mean)
    label = key.replace("_cycles1","").replace("steps","") + " steps/task"
    ax_c1.plot(range(n_pts), mean, color=col, lw=2, label=label)
    ax_c1.fill_between(range(n_pts), mean-std, mean+std, color=col, alpha=0.18)
ax_c1.axhline(0, color="black", lw=0.8, ls="--")
ax_c1.set_xlabel("Task presentation index"); ax_c1.set_ylabel("LoP gap")
ax_c1.set_title("Steps sweep  (mean ± std, 3 seeds)", fontsize=9)
ax_c1.legend(fontsize=8, frameon=False)
ax_c1.set_xticks(range(4))
ax_c1.set_xticklabels([f"t{i}\n{TASK_ANGLES[i]}°" for i in range(4)])

# Panel 2: per-seed bars for each steps condition
n_conditions = len(step_keys)
bar_w = 0.22
for ci, (col, key) in enumerate(zip(colors_s, step_keys)):
    cond = data["steps_sweep"].get(key)
    if cond is None: continue
    seed_colors = [col] * len(cond["seeds"])
    for si, seed_r in enumerate(cond["seeds"]):
        lops = [b["lop_gap"] for b in seed_r["block_logs"]]
        offset = (ci - 1) * bar_w
        bars = ax_c2.bar(np.arange(4) + offset, lops, width=bar_w,
                         color=col, alpha=0.3 + 0.14*si,
                         label=f"{key.replace('_cycles1','').replace('steps','')}s s{si}" if si==0 else "")
ax_c2.axhline(0, color="black", lw=0.8, ls="--")
ax_c2.set_xlabel("Task presentation index"); ax_c2.set_ylabel("LoP gap")
ax_c2.set_title("Per-seed LoP by steps condition", fontsize=9)
ax_c2.set_xticks(range(4))
ax_c2.set_xticklabels([f"t{i}\n{TASK_ANGLES[i]}°" for i in range(4)])
ax_c2.legend(fontsize=7, frameon=False)

plt.suptitle("Figure C — LoP gap: steps sweep summary\n"
             "positive = prefix model learns slower than fresh (lost plasticity)",
             fontweight="bold")
plt.tight_layout()
out_c = os.path.join(RESULTS, "rotating_target", "figC_lop.png")
fig_c.savefig(out_c, dpi=150, bbox_inches="tight")
print(f"\nsaved {out_c}")
plt.close(fig_c)

print("\nAll figures saved.")
