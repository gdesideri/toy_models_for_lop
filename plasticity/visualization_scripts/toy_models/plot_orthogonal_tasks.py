"""
plot_orthogonal_tasks.py — Figures of experiments/run_orthogonal_tasks.py.

Usage (from plasticity/)
    uv run python -m visualization_scripts.toy_models.plot_orthogonal_tasks --run-dir results/orthogonal_tasks/<run>
        [--seeds 0 1 ...]   (default: every seed in the folder)
        [--planes 0,1 0,2]  (weight/input planes for D > 2, 0-based axes)
        [--out <dir>]       (default: <run>/figures)

Figures (input space and weight space are always labelled as such)
    input_space.png               input clouds of both conditions, support outline, task boundaries
                                  (first seed; D > 2: projections onto coordinate planes)
    seed<s>_weights.png           first-layer weight vectors w_k over training, one colour per unit,
                                  task-switch markers; D = 2 exact with the inactive cone,
                                  D = 3 3-D view plus planar projections, D > 3 projections only
    seed<s>_cone.png              the inactive cone: unit directions vs step (D = 2, polar), the
                                  normalised support maximum m(w)/||w|| (exact in any D; <= 0 is
                                  certified inactive) and the activity on the train sample
    seed<s>_units.png             time series per unit: direction, norm, readout, activation
                                  frequency, full-batch gradient, rotation / radial parts of the
                                  SGD update, output contribution and cancellation index
    seed<s>_learning.png          held-out accuracy and loss, continued vs fresh, and live units
    aggregate_units.png           unit fates of every seed (no selection), continued and fresh
    aggregate_learning.png        fresh − continued final accuracy per phase, fresh solvability

Projections never decide inactivity: every status shown is computed in the full input space.
Weight trajectories are not clipped (axes cover the whole path) and near-zero vectors get no angle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from experiments.run_orthogonal_tasks import MECHANISMS, seed_for  # noqa: E402
from src import unit_geometry as geo  # noqa: E402
from tasks.orthogonal_halfspaces import build_tasks  # noqa: E402
from visualization_scripts.common import save, set_style  # noqa: E402

UNIT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
TASK_TINTS = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00"]
LABEL_COLORS = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00"]          # (y_i, y_j) = 00, 01, 10, 11
CONE = "#7f7f7f"
CONTINUED, FRESH = "#000000", "#D55E00"
MECH_COLORS = {"initially_inactive": "#7f7f7f", "rotation_into_cone": "#000000",
               "rotation_to_cone_edge": "#D55E00", "norm_collapse": "#CC79A7", "bias_driven": "#E69F00",
               "live": "#009E73"}
COND_TITLE = {"shifted": "shifted", "centered": "centered control"}


# ── Loading ───────────────────────────────────────────────────────────────────

def seeds_in(run_dir: Path) -> list[int]:
    return sorted(int(p.parent.name[4:]) for p in run_dir.glob("seed*/summary.json"))


def load_seed(run_dir: Path, seed: int):
    sd = run_dir / f"seed{seed}"
    summary = json.loads((sd / "summary.json").read_text())
    arrays = {c: dict(np.load(sd / f"{c}.npz")) for c in summary["conditions"]}
    return summary["config"], summary, arrays


def unit_colors(h: int) -> list:
    if h <= len(UNIT_COLORS):
        return UNIT_COLORS[:h]
    return [plt.get_cmap("tab20")(i % 20) for i in range(h)]


def task_name(j: int) -> str:
    return chr(ord("A") + j) if j < 26 else str(j)


def fmt_vec(v) -> str:
    return "(" + ", ".join(f"{x:g}" for x in np.round(v, 3)) + ")"


def cond_label(cond: str, mu) -> str:
    return f"{COND_TITLE[cond]}, μ = {fmt_vec(mu)}"


def default_planes(d: int, k: int, mu) -> list[tuple[int, int]]:
    """Coordinate planes to project onto: consecutive task axes, plus one through the shift axis."""
    if d == 2:
        return [(0, 1)]
    planes = [(i, i + 1) for i in range(min(k, d) - 1)][:3] or [(0, 1)]
    mu = np.asarray(mu)
    if np.any(mu != 0):
        m = int(np.argmax(np.abs(mu)))
        if not any(m in p for p in planes):
            planes.append((0, m) if m != 0 else (0, 1))
    return planes[:4]


def grid(n_conds: int, n_panels: int):
    """(rows, cols, index(r, c)) of a figure with one row per condition, or one row in all when a
    condition has a single panel (then conditions sit side by side)."""
    if n_panels == 1:
        return 1, n_conds, lambda r, c: r + 1
    return n_conds, n_panels, lambda r, c: r * n_panels + c + 1


def death_steps(summary_cond: dict) -> dict[int, dict]:
    return {u["unit"]: u["death"] for u in summary_cond["units"] if u["death"]}


def log_index(steps, step) -> int:
    """First logged state at or after `step`."""
    i = np.searchsorted(steps, step)
    return int(min(i, len(steps) - 1))


def shade_tasks(ax, schedule, S, vlines=True) -> None:
    for p, task in enumerate(schedule):
        ax.axvspan(p * S, (p + 1) * S, color=TASK_TINTS[task % len(TASK_TINTS)], alpha=0.08, lw=0)
        if vlines and p:
            ax.axvline(p * S, color="grey", lw=0.6, ls="--", alpha=0.6)


def task_legend_handles(k: int) -> list:
    return [Patch(color=TASK_TINTS[j % len(TASK_TINTS)], alpha=0.25, label=f"training on task {task_name(j)}")
            for j in range(k)]


def cone_wedges_2d(ax, mu, A, radius, **kw) -> bool:
    """Shade the certified inactive cone (b = 0) of the weight plane; False when it is empty."""
    arcs = geo.inactive_arcs_2d(mu, A)
    for lo, hi in arcs:
        span = (hi - lo) % 360 or 360
        th = np.radians(np.linspace(lo, lo + span, 200))
        pts = np.vstack([[0, 0], np.column_stack([np.cos(th), np.sin(th)]) * radius])
        ax.add_patch(Polygon(pts, closed=True, **kw))
    return bool(arcs)


def break_wraps(angle_deg):
    """Insert NaN where a ±180° angle wraps, so lines do not jump across the axis."""
    a = np.array(angle_deg, float)
    jump = np.abs(np.diff(a, axis=0)) > 180
    a[1:][jump] = np.nan
    return a


# ── Figure 1: input space ─────────────────────────────────────────────────────

def plot_input_space(cfg: dict, seed: int, planes, out: Path) -> None:
    k, d = cfg["n_tasks"], cfg["input_dim"]
    tasks = build_tasks(k, d, cfg["n_train"], cfg["n_test"], seed_for(seed, "data"), mu=cfg["mu"],
                        scales=cfg["scales"])
    conds = cfg["conditions"]
    nr, nc, at = grid(len(conds), len(planes))
    fig = plt.figure(figsize=(5.6 * nc, 5.0 * nr + 0.9))
    rng = np.random.default_rng(0)
    show = rng.permutation(tasks.u_train.shape[0])[:1500]
    cov = tasks.A @ tasks.A.T
    t = np.linspace(0, 2 * np.pi, 400)
    for r, cond in enumerate(conds):
        mu = tasks.condition_mu(cond)
        x = tasks.inputs("train", cond)[show]
        y = tasks.labels_train[:, show]
        for c, (i, j) in enumerate(planes):
            ax = fig.add_subplot(nr, nc, at(r, c))
            if i < k and j < k:
                code = 2 * y[i] + y[j]
                for v in range(4):
                    m = code == v
                    ax.scatter(x[m, i], x[m, j], s=5, color=LABEL_COLORS[v], alpha=0.6, lw=0,
                               label=f"y_{task_name(i)}={v // 2}, y_{task_name(j)}={v % 2}")
            elif i < k:
                for v in range(2):
                    m = y[i] == v
                    ax.scatter(x[m, i], x[m, j], s=5, color=LABEL_COLORS[2 * v], alpha=0.6, lw=0,
                               label=f"y_{task_name(i)}={v}")
            else:
                ax.scatter(x[:, i], x[:, j], s=5, color="grey", alpha=0.5, lw=0)
            # exact outline of the projected support ellipsoid
            L = np.linalg.cholesky(cov[np.ix_([i, j], [i, j])])
            edge = mu[[i, j]] + (L @ np.vstack([np.cos(t), np.sin(t)])).T
            ax.plot(edge[:, 0], edge[:, 1], color="k", lw=1.0, label="support boundary")
            for tj, (axis, style) in enumerate(((i, "-"), (j, "--"))):
                if axis < k:
                    if tj == 0:
                        ax.axvline(mu[i], color="k", ls=style, lw=1.4, label=f"task {task_name(i)} boundary")
                        ax.annotate("", xy=(mu[i] + 0.35, mu[j]), xytext=(mu[i], mu[j]),
                                    arrowprops=dict(arrowstyle="->", lw=1.4))
                        ax.text(mu[i] + 0.37, mu[j], f"$v_{task_name(i)}$", va="center")
                    else:
                        ax.axhline(mu[j], color="k", ls=style, lw=1.4, label=f"task {task_name(j)} boundary")
                        ax.annotate("", xy=(mu[i], mu[j] + 0.35), xytext=(mu[i], mu[j]),
                                    arrowprops=dict(arrowstyle="->", lw=1.4))
                        ax.text(mu[i], mu[j] + 0.38, f"$v_{task_name(j)}$", ha="center")
            ax.plot(*mu[[i, j]], marker="*", color="k", ms=10, ls="", label="μ")
            ax.plot(0, 0, marker="P", color="#D55E00", ms=8, ls="",
                    label="origin: every bias-free unit boundary w·x = 0 passes here")
            ax.set_aspect("equal")
            ax.set_xlabel(f"input $x_{i + 1}$")
            ax.set_ylabel(f"input $x_{j + 1}$")
            proj = "" if d == 2 else f" — projection onto ($x_{i + 1}$, $x_{j + 1}$)"
            ax.set_title(f"Input space, {cond_label(cond, mu)}{proj}", fontsize=10)
            lo = np.minimum(edge.min(0), 0) - 0.3
            hi = np.maximum(edge.max(0), 0) + 0.5
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            if r == 0 and c == 0:
                handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(f"Input space (seed {seed}): one train set, only labels change between tasks", fontsize=12)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, markerscale=1.0, frameon=False)
    fig.tight_layout(rect=(0, 0.9 / (5.0 * nr + 0.9) + 0.02, 1, 1))
    save(fig, out / "input_space.png")


# ── Figure 2: weight trajectories ─────────────────────────────────────────────

def _trajectory_markers(ax, P, steps, schedule, S, deaths, h, color, init_dead, z=None):
    """Start / switch / end / death markers of unit h on 2-D (or 3-D with z) coordinates P."""
    def pt(i):
        return (P[i, 0], P[i, 1]) if z is None else (P[i, 0], P[i, 1], z[i])
    if not init_dead:
        ax.plot(*pt(len(steps) - 1), marker="s", ms=6, color=color, ls="")
        for p in range(1, len(schedule)):
            i = log_index(steps, p * S)
            ax.plot(*pt(i), marker="D", ms=5, mfc="white", mec=color, mew=1.2, ls="")
    ax.plot(*pt(0), marker="o", ms=8, color=color, mfc="white" if init_dead else color, mew=2, ls="")
    if h in deaths:
        ax.plot(*pt(log_index(steps, deaths[h]["step"])), marker="X", ms=11, color="k", mec="white", ls="")


def marker_legend(two_d_cone: bool) -> list:
    handles = [Line2D([], [], marker="o", color="k", ls="", label="start"),
               Line2D([], [], marker="o", color="k", mfc="white", ls="",
                      label="start, silent at init (never moves)"),
               Line2D([], [], marker="D", mfc="white", mec="k", ls="", label="state at a task switch"),
               Line2D([], [], marker="s", color="k", ls="", label="end"),
               Line2D([], [], marker="X", color="k", ls="", ms=9, label="unit lost its training signal")]
    if two_d_cone:
        handles.append(Patch(color=CONE, alpha=0.3, label="certified inactive cone (b = 0)"))
    return handles


def plot_weights(cfg, summary, arrays, seed, planes, out: Path) -> None:
    d, conds = cfg["input_dim"], list(arrays)
    H = cfg["width"]
    colors = unit_colors(H)
    three_d = d == 3
    nr, nc, at = grid(len(conds), len(planes) + (1 if three_d else 0))
    fig = plt.figure(figsize=(5.6 * nc, 5.4 * nr))
    for r, cond in enumerate(conds):
        arr, sc = arrays[cond], summary["conditions"][cond]
        W, steps = arr["W"], arr["step"]
        schedule, S = arr["schedule"].tolist(), int(arr["steps_per_task"])
        dead = death_steps(sc)
        init_dead = (arr["init_status"] != geo.ACTIVE) | arr["init_silent_train"]
        lim = 1.15 * max(np.abs(W).max(), 1e-3)
        col = 0
        if three_d:
            ax = fig.add_subplot(nr, nc, at(r, 0), projection="3d")
            # inactive directions on a sphere of radius lim (full-space criterion)
            th, ph = np.meshgrid(np.linspace(0, np.pi, 60), np.linspace(0, 2 * np.pi, 120))
            dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
            dead_dir = geo.certified_inactive(dirs, 0.0, arr["mu"], arr["A"])
            fc = np.zeros(dirs.shape[:2] + (4,))
            fc[dead_dir] = matplotlib.colors.to_rgba(CONE, 0.35)
            if dead_dir.any():
                ax.plot_surface(*(lim * dirs).transpose(2, 0, 1), facecolors=fc, shade=False, lw=0)
            for h in range(H):
                ax.plot(W[:, h, 0], W[:, h, 1], W[:, h, 2], color=colors[h], lw=1.3)
                _trajectory_markers(ax, W[:, h, :2], steps, schedule, S, dead, h, colors[h], init_dead[h],
                                    z=W[:, h, 2])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_xlabel("$w_1$")
            ax.set_ylabel("$w_2$")
            ax.set_zlabel("$w_3$")
            ax.set_title(f"Weight space (3-D), {cond_label(cond, arr['mu'])}\n"
                         f"grey: certified inactive directions (b = 0), drawn at radius {lim:.2f}", fontsize=9)
            col = 1
        for c, (i, j) in enumerate(planes):
            ax = fig.add_subplot(nr, nc, at(r, col + c))
            if d == 2:
                has_cone = cone_wedges_2d(ax, arr["mu"], arr["A"], 5 * lim, color=CONE, alpha=0.3, lw=0)
            else:
                has_cone = False
            for t, vt in enumerate(arr["normals"]):
                pi, pj = vt[i], vt[j]
                if abs(pi) + abs(pj) > 0:
                    ax.plot([0, 0.9 * lim * pi], [0, 0.9 * lim * pj], color="grey", lw=0.8, ls=":")
                    ax.text(0.93 * lim * pi, 0.93 * lim * pj, f"$v_{task_name(t)}$", color="grey",
                            ha="center", va="center", fontsize=9)
            for h in range(H):
                ax.plot(W[:, h, i], W[:, h, j], color=colors[h], lw=1.3, label=f"unit {h}")
                _trajectory_markers(ax, W[:, h][:, [i, j]], steps, schedule, S, dead, h, colors[h], init_dead[h])
            ax.axhline(0, color="k", lw=0.4)
            ax.axvline(0, color="k", lw=0.4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"weight on $x_{i + 1}$:  $w_{i + 1}$")
            ax.set_ylabel(f"weight on $x_{j + 1}$:  $w_{j + 1}$")
            if d == 2:
                note = "exact" + ("; grey: certified inactive cone (b = 0)" if has_cone
                                  else "; no nonzero inactive direction")
            else:
                note = f"projection onto ($w_{i + 1}$, $w_{j + 1}$): NOT an inactivity test"
            ax.set_title(f"Weight space, {cond_label(cond, arr['mu'])}\n{note}", fontsize=9)
            if r == 0 and c == 0:
                leg1 = ax.legend(fontsize=7, loc="upper left")
                ax.add_artist(leg1)
                ax.legend(handles=marker_legend(d == 2), fontsize=7, loc="lower right", framealpha=0.85)
    fig.suptitle(f"First-layer weight vectors $w_k$ over training (seed {seed}), task order "
                 f"{', '.join(task_name(t) for t in arrays[conds[0]]['schedule'])}",
                 fontsize=11)
    fig.tight_layout()
    save(fig, out / f"seed{seed}_weights.png")


# ── Figure 3: the inactive cone ───────────────────────────────────────────────

def plot_cone(cfg, summary, arrays, seed, out: Path) -> None:
    d, conds, H = cfg["input_dim"], list(arrays), cfg["width"]
    colors = unit_colors(H)
    ncols = 3 if d == 2 else 2
    fig = plt.figure(figsize=(5.6 * ncols, 4.6 * len(conds)))
    for r, cond in enumerate(conds):
        arr, sc = arrays[cond], summary["conditions"][cond]
        steps, schedule, S = arr["step"], arr["schedule"].tolist(), int(arr["steps_per_task"])
        dead = death_steps(sc)
        norm = arr["norm"]
        # normalised support max m(w)/||w|| = ŵ·μ + ||Aᵀŵ|| + b/||w||, NaN for collapsed units
        with np.errstate(invalid="ignore", divide="ignore"):
            nm = np.where(norm < float(arr["norm_eps"]), np.nan, arr["support_max"] / norm)
        c0 = 0
        if d == 2:
            ax = fig.add_subplot(len(conds), ncols, r * ncols + 1, projection="polar")
            T = steps[-1]
            for lo, hi in geo.inactive_arcs_2d(arr["mu"], arr["A"]):
                span = (hi - lo) % 360 or 360
                ax.fill_between(np.radians(np.linspace(lo, lo + span, 100)), 0, T, color=CONE, alpha=0.3, lw=0)
            for h in range(H):
                ang = np.radians(arr["angle_2d"][:, h])
                ax.plot(ang, steps, color=colors[h], lw=1.2, label=f"unit {h}")
                for p in range(1, len(schedule)):
                    i = log_index(steps, p * S)
                    ax.plot(ang[i], steps[i], marker="D", ms=4, mfc="white", mec=colors[h], ls="")
                if h in dead:
                    i = log_index(steps, dead[h]["step"])
                    ax.plot(ang[i], steps[i], marker="X", ms=10, color="k", mec="white", ls="")
            ax.set_rlim(0, T)
            ax.set_yticks([p * S for p in range(1, len(schedule) + 1)])
            ax.set_yticklabels([])
            ax.text(np.radians(160), T * 1.18, f"radius: step 0 (centre) to {T}\nrings: task switches",
                    fontsize=7, ha="center")
            ax.set_title(f"Direction of $w_k$ (angle) vs training step (radius)\nweight space, "
                         f"{cond_label(cond, arr['mu'])}; grey: inactive cone", fontsize=9)
            if r == 0:
                ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(-0.25, 1.1))
            c0 = 1
        ax = fig.add_subplot(len(conds), ncols, r * ncols + c0 + 1)
        shade_tasks(ax, schedule, S)
        ax.axhspan(-10, 0, color=CONE, alpha=0.25, lw=0)
        ax.axhline(0, color="k", lw=0.8)
        for h in range(H):
            ax.plot(steps, nm[:, h], color=colors[h], lw=1.2)
            if h in dead:
                ax.plot(dead[h]["step"], nm[log_index(steps, dead[h]["step"]), h], marker="X", ms=9, color="k",
                        mec="white", ls="")
        ax.set_yscale("symlog", linthresh=0.01)
        lo = min(-0.05, np.nanmin(nm) * 1.2) if np.isfinite(np.nanmin(nm)) else -0.05
        ax.set_ylim(lo, np.nanmax(nm) * 1.3 if np.isfinite(np.nanmax(nm)) else 2)
        ax.set_xlabel("training step")
        ax.set_ylabel("max over support of $w\\cdot x + b$, / $\\|w\\|$")
        ax.set_title(f"Normalised support maximum (exact, full space), {COND_TITLE[cond]}\n"
                     "grey (≤ 0): certified inactive; symlog, linear within ±0.01", fontsize=9)
        ax = fig.add_subplot(len(conds), ncols, r * ncols + c0 + 2)
        shade_tasks(ax, schedule, S)
        n_train = cfg["n_train"]
        floor = 0.5 / n_train
        for h in range(H):
            a = arr["act_frac_train"][:, h]
            ax.plot(steps, np.where(a > 0, a, floor), color=colors[h], lw=1.2)
        ax.axhline(floor, color="k", lw=0.6, ls=":")
        ax.text(steps[-1], floor * 1.3, "0: silent on the whole train sample", ha="right", va="bottom", fontsize=7)
        ax.set_yscale("log")
        ax.set_ylim(floor / 2, 1.2)
        ax.set_xlabel("training step")
        ax.set_ylabel("fraction of train inputs with $w\\cdot x + b > 0$")
        ax.set_title(f"Activity on the finite train sample (n = {n_train}), {COND_TITLE[cond]}\n"
                     "silent ≠ certified inactive: compare with the middle panel", fontsize=9)
    fig.suptitle(f"Inactive cone (seed {seed}): units entering or avoiding it", fontsize=11)
    fig.tight_layout()
    save(fig, out / f"seed{seed}_cone.png")


# ── Figure 4: unit time series ────────────────────────────────────────────────

def plot_units(cfg, summary, arrays, seed, out: Path) -> None:
    d, conds, H = cfg["input_dim"], list(arrays), cfg["width"]
    colors = unit_colors(H)
    rows = ["direction", "norm", "readout", "activity", "grad", "rotation", "radial", "contrib"]
    fig, axes = plt.subplots(len(rows), len(conds), figsize=(7.5 * len(conds), 2.3 * len(rows)),
                             sharex="col", squeeze=False)
    for c, cond in enumerate(conds):
        arr, sc = arrays[cond], summary["conditions"][cond]
        steps, schedule, S = arr["step"], arr["schedule"].tolist(), int(arr["steps_per_task"])
        eps = float(arr["norm_eps"])
        dead = death_steps(sc)
        task_now = arr["task"]
        ax_of = {name: axes[i, c] for i, name in enumerate(rows)}
        for ax in axes[:, c]:
            shade_tasks(ax, schedule, S)
        for h in range(H):
            kw = dict(color=colors[h], lw=1.1)
            if d == 2:
                ax_of["direction"].plot(steps, break_wraps(arr["angle_2d"][:, h]), label=f"unit {h}", **kw)
            else:
                ax_of["direction"].plot(steps, arr["angle_to_task_normal"][:, h], label=f"unit {h}", **kw)
            ax_of["norm"].plot(steps, arr["norm"][:, h], **kw)
            ax_of["readout"].plot(steps, arr["a"][:, h], **kw)
            ax_of["activity"].plot(steps, arr["act_frac_train"][:, h], **kw)
            g = arr["grad_w_norm"][np.arange(len(steps)), task_now, h]
            ax_of["grad"].plot(steps, g, **kw)
            ax_of["rotation"].plot(steps, arr["update_w_tangential"][:, h], **kw)
            ax_of["radial"].plot(steps, arr["update_w_radial"][:, h], **kw)
            ax_of["contrib"].plot(steps, arr["contrib_std"][:, h], **kw)
            if h in dead:
                for ax in axes[:, c]:
                    ax.axvline(dead[h]["step"], color=colors[h], lw=1.0, ls=":")
        if d == 2:
            for lo, hi in geo.inactive_arcs_2d(arr["mu"], arr["A"]):
                if lo <= hi:
                    ax_of["direction"].axhspan(lo, hi, color=CONE, alpha=0.3, lw=0)
                else:
                    ax_of["direction"].axhspan(lo, 180, color=CONE, alpha=0.3, lw=0)
                    ax_of["direction"].axhspan(-180, hi, color=CONE, alpha=0.3, lw=0)
            ax_of["direction"].set_ylim(-180, 180)
            ax_of["direction"].set_yticks([-180, -90, 0, 90, 180])
            ax_of["direction"].set_ylabel("signed angle of $w_k$\natan2($w_2$, $w_1$) [°]")
        else:
            ax_of["direction"].set_ylim(0, 180)
            ax_of["direction"].set_ylabel("angle to current\ntask normal [°]")
        ax_of["norm"].set_yscale("log")
        ax_of["norm"].axhline(eps, color="k", lw=0.6, ls=":")
        ax_of["norm"].set_ylabel("$\\|w_k\\|$ (log)\n(dotted: collapse eps)")
        ax_of["readout"].axhline(0, color="k", lw=0.6)
        ax_of["readout"].set_ylabel("output weight $a_k$")
        ax_of["activity"].set_ylim(-0.03, 1.03)
        ax_of["activity"].set_ylabel("activation freq.\n(train sample)")
        for name in ("grad", "rotation"):
            ax_of[name].set_yscale("symlog", linthresh=1e-6)
            ax_of[name].set_ylim(bottom=0)
        ax_of["grad"].set_ylabel("full-batch $\\|\\partial L/\\partial w_k\\|$\ncurrent task (symlog)")
        ax_of["rotation"].set_ylabel("rotation part of\nlast SGD update (symlog)")
        ax_of["radial"].set_yscale("symlog", linthresh=1e-6)
        ax_of["radial"].axhline(0, color="k", lw=0.6)
        ax_of["radial"].set_ylabel("radial part of\nlast update (+: grows)")
        ax_of["contrib"].set_ylabel("std of $a_k\\,$relu($w_k\\cdot x$)\n(test); black: cancel.")
        tw = ax_of["contrib"].twinx()
        tw.plot(steps, arr["cancellation"], color="k", lw=0.9)
        tw.set_ylabel("cancellation index\nΣ std(c_k) / std(Σ c_k)", fontsize=8)
        axes[-1, c].set_xlabel("training step")
        axes[0, c].set_title(f"Per-unit time series, {cond_label(cond, arr['mu'])}\n"
                             "(dotted verticals: a unit lost its training signal)", fontsize=10)
        if c == 0:
            h1 = [Line2D([], [], color=colors[h], label=f"unit {h}") for h in range(H)]
            axes[0, c].legend(handles=h1 + task_legend_handles(cfg["n_tasks"]), fontsize=7, ncol=2, loc="best")
    fig.suptitle(f"Hidden units over training (seed {seed}); grey band in row 1: inactive cone", fontsize=11)
    fig.tight_layout()
    save(fig, out / f"seed{seed}_units.png")


# ── Figure 5: learning curves ─────────────────────────────────────────────────

def live_count(status, act):
    return ((status == geo.ACTIVE) & (act > 0)).sum(-1)


def plot_learning(cfg, summary, arrays, seed, out: Path) -> None:
    conds, k = list(arrays), cfg["n_tasks"]
    fig, axes = plt.subplots(4, len(conds), figsize=(7.5 * len(conds), 10), sharex="col", squeeze=False)
    thr = cfg["solve_threshold"]
    for c, cond in enumerate(conds):
        arr, sc = arrays[cond], summary["conditions"][cond]
        steps, schedule, S = arr["step"], arr["schedule"].tolist(), int(arr["steps_per_task"])
        idx = np.arange(len(steps))
        for ax in axes[:, c]:
            shade_tasks(ax, schedule, S)
        cur = arr["test_acc"][idx, arr["task"]]
        axes[0, c].plot(steps, cur, color=CONTINUED, lw=1.3, label="continued net")
        for p in range(len(schedule)):
            m = arr["fresh_phase"] == p
            fs, task = arr["fresh_step"][m], schedule[p]
            axes[0, c].plot(fs, arr["fresh_test_acc"][m, task], color=FRESH, lw=1.1, ls="--",
                            label="fresh net (same minibatches)" if p == 0 else None)
            axes[2, c].plot(fs, arr["fresh_test_loss"][m, task], color=FRESH, lw=1.1, ls="--")
            axes[3, c].plot(fs, live_count(arr["fresh_status"][m], arr["fresh_act_frac_train"][m]), color=FRESH,
                            lw=1.1, ls="--")
            ph = sc["phases"][p]
            if not ph["fresh_solved"]:
                axes[0, c].text((p + 0.5) * S, 0.52, "fresh\nnot solved", ha="center", fontsize=7, color=FRESH)
        axes[0, c].axhline(thr, color="grey", lw=0.7, ls=":")
        axes[0, c].set_ylim(0.4, 1.01)
        axes[0, c].set_ylabel("held-out accuracy\non the current task")
        for j in range(k):
            axes[1, c].plot(steps, arr["test_acc"][:, j], color=TASK_TINTS[j % len(TASK_TINTS)], lw=1.2,
                            label=f"task {task_name(j)}")
        axes[1, c].set_ylim(0.4, 1.01)
        axes[1, c].set_ylabel("continued net: held-out\naccuracy on every task")
        axes[1, c].legend(fontsize=7, loc="lower left")
        axes[2, c].plot(steps, arr["test_loss"][idx, arr["task"]], color=CONTINUED, lw=1.3)
        axes[2, c].set_yscale("log")
        axes[2, c].set_ylabel("held-out BCE\non the current task")
        axes[3, c].plot(steps, live_count(arr["status"], arr["act_frac_train"]), color=CONTINUED, lw=1.3)
        axes[3, c].set_ylabel("live units\n(active on train sample)")
        axes[3, c].set_ylim(-0.2, cfg["width"] + 0.2)
        axes[3, c].set_xlabel("training step")
        axes[0, c].set_title(f"Continued vs fresh, {cond_label(cond, arr['mu'])}", fontsize=10)
        if c == 0:
            axes[0, c].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Held-out learning curves (seed {seed}); dotted: solvability threshold {thr}", fontsize=11)
    fig.tight_layout()
    save(fig, out / f"seed{seed}_learning.png")


# ── Aggregates over every seed ────────────────────────────────────────────────

def unit_fates(sc: dict) -> dict:
    fates = {k: 0 for k in MECH_COLORS}
    for u in sc["units"]:
        if u["init_status"] != "active" or u["init_silent_on_train_sample"]:
            fates["initially_inactive"] += 1
        elif u["death"]:
            fates[u["death"]["mechanism"]] += 1
        else:
            fates["live"] += 1
    return fates


def fresh_fates(sc: dict) -> dict:
    c = sc["counts"]
    # certified inactive at init implies silent on the sample, so this counts both
    fates = {"initially_inactive": c["fresh_initially_silent_on_train_sample"]}
    for m in MECHANISMS:
        fates[m] = c[f"fresh_died_{m}"]
    fates["live"] = sum(ph["fresh_live_end"] for ph in sc["phases"])
    return fates


def plot_aggregate(run_dir: Path, seeds, out: Path) -> None:
    summaries = [json.loads((run_dir / f"seed{s}" / "summary.json").read_text()) for s in seeds]
    conds = list(summaries[0]["conditions"])
    fig, axes = plt.subplots(2, len(conds), figsize=(6.5 * len(conds), 7.5), squeeze=False)
    for c, cond in enumerate(conds):
        for r, fates_of in enumerate((unit_fates, fresh_fates)):
            ax = axes[r, c]
            bottom = np.zeros(len(seeds))
            for key, color in MECH_COLORS.items():
                vals = np.array([fates_of(s["conditions"][cond]).get(key, 0) for s in summaries], float)
                ax.bar(np.arange(len(seeds)), vals, bottom=bottom, color=color, label=key.replace("_", " "))
                bottom += vals
            ax.set_xticks(np.arange(len(seeds)), [str(s) for s in seeds])
            ax.set_xlabel("seed")
            if r == 0:
                ax.set_ylabel("hidden units of the continued net")
                ax.set_title(f"Continued net: fate of each unit, {COND_TITLE[cond]}", fontsize=10)
            else:
                ax.set_ylabel("unit-phases of the fresh nets")
                ax.set_title(f"Fresh nets (all phases pooled), {COND_TITLE[cond]}", fontsize=10)
    axes[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"Unit fates over all {len(seeds)} seeds: lost training signal by mechanism", fontsize=11)
    fig.tight_layout()
    save(fig, out / "aggregate_units.png")

    fig, axes = plt.subplots(2, len(conds), figsize=(6.5 * len(conds), 7), squeeze=False, sharex=True)
    rng = np.random.default_rng(0)
    for c, cond in enumerate(conds):
        for s, summ in enumerate(summaries):
            ph = summ["conditions"][cond]["phases"]
            x = np.arange(len(ph)) + rng.uniform(-0.15, 0.15)
            gap = np.array([p["fresh_minus_continued_acc"] for p in ph])
            solved = np.array([p["fresh_solved"] for p in ph])
            axes[0, c].scatter(x[solved], gap[solved], color="k", s=14)
            axes[0, c].scatter(x[~solved], gap[~solved], facecolor="white", edgecolor="k", s=14)
            axes[1, c].scatter(x, [p["fresh_final_test_acc"] for p in ph], color=FRESH, s=12)
            axes[1, c].scatter(x, [p["continued_final_test_acc"] for p in ph], color=CONTINUED, s=12, marker="x")
        sched = summaries[0]["conditions"][cond]["phases"]
        axes[0, c].axhline(0, color="k", lw=0.6)
        axes[0, c].set_ylabel("fresh − continued\nfinal held-out accuracy")
        axes[0, c].set_title(f"{COND_TITLE[cond]}: loss of plasticity per phase\n(hollow: fresh net did not "
                             "solve the task, gap not interpretable)", fontsize=9)
        axes[1, c].axhline(summaries[0]["config"]["solve_threshold"], color="grey", ls=":", lw=0.8)
        axes[1, c].set_ylabel("final held-out accuracy")
        axes[1, c].legend(handles=[Line2D([], [], marker="o", color=FRESH, ls="", label="fresh"),
                                   Line2D([], [], marker="x", color=CONTINUED, ls="", label="continued")],
                          fontsize=7, loc="lower left")
        many = len(sched) > 12
        axes[1, c].set_xticks(np.arange(len(sched)), [f"{p['phase']}:{task_name(p['task'])}" for p in sched],
                              rotation=90 if many else 0, fontsize=8 if many else None)
        axes[1, c].set_xlabel("phase:task")
    fig.suptitle(f"Continued vs fresh over all {len(seeds)} seeds", fontsize=11)
    fig.tight_layout()
    save(fig, out / "aggregate_learning.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", "--run_dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--planes", nargs="+", default=None, help="0-based axis pairs such as 0,1 0,2")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    set_style()
    seeds = args.seeds or seeds_in(args.run_dir)
    if not seeds:
        raise SystemExit(f"no seed*/summary.json in {args.run_dir}")
    out = args.out or args.run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for i, seed in enumerate(seeds):
        cfg, summary, arrays = load_seed(args.run_dir, seed)
        planes = ([tuple(int(v) for v in s.split(",")) for s in args.planes] if args.planes
                  else default_planes(cfg["input_dim"], cfg["n_tasks"], cfg["mu"]))
        if i == 0:
            plot_input_space(cfg, seed, planes, out)
        plot_weights(cfg, summary, arrays, seed, planes, out)
        plot_cone(cfg, summary, arrays, seed, out)
        plot_units(cfg, summary, arrays, seed, out)
        plot_learning(cfg, summary, arrays, seed, out)
    plot_aggregate(args.run_dir, seeds, out)


if __name__ == "__main__":
    main()
