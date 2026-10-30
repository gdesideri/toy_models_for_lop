"""
animate_orthogonal_tasks.py — GIF of the hidden units of one seed of an orthogonal_tasks run (D = 2 or 3).

Three panels per frame:
    input space    train inputs coloured by the label of the current task; the task boundary (black dashed:
                   one line for a half-plane, both edges of a stripe, both lines of an XOR); each unit's own
                   boundary w·x + b = 0 in its colour with an arrow on its active side, dotted once the unit is
                   silent on every train input; in D = 2 also the net's decision boundary (solid black, where
                   the logit is 0) over a background shaded by the predicted class
    weight space   each unit's weight vector w with its trail so far; an X where it died
    accuracy       held-out accuracy on the current task: continued net vs a fresh net per phase
In D = 3 the first two panels are 3-D views that turn slowly (task boundary and unit boundaries are planes).

Usage (from plasticity/)
    uv run python -m visualization_scripts.toy_models.animate_orthogonal_tasks --run-dir results/orthogonal_tasks/<run> \
        --seed 0 [--every 100] [--window 1500] [--coarse 500] [--fps 12] [--dpi 70] [--spin 0.6] [--out <file.gif>]
Frames are dense (every --every steps) for the first --window steps after each switch, every --coarse
steps otherwise. Default output: <run>/figures/seed<s>_units.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from orthogonal_tasks.config import task_name  # noqa: E402
from orthogonal_tasks.summary import load_seed  # noqa: E402
from orthogonal_tasks.train import build_seed_tasks  # noqa: E402
from visualization_scripts.common import set_style  # noqa: E402
from visualization_scripts.toy_models.orthogonal_common import (CONTINUED, FRESH, LABEL0, LABEL1,  # noqa: E402
                                                                log_index, shade_tasks, unit_colors)


def frame_steps(total: int, every: int, window: int, coarse: int, n_phases: int, S: int) -> np.ndarray:
    fine = [np.arange(q * S, min(q * S + window, total) + 1, every) for q in range(n_phases)]
    return np.unique(np.concatenate(fine + [np.arange(0, total + 1, coarse), np.arange(n_phases + 1) * S]))


def unit_label(h, a, b, act) -> str:
    return f"unit {h}: a={a:+.1f}, b={b:+.1f}, active {act:.0%}"


# ── D = 2 ──────────────────────────────────────────────────────────────────────

def draw_input_2d(ax, x, labels, A, boundaries, W, b, a, c, silent, act, colors):
    t = np.linspace(0, 2 * np.pi, 200)
    edge = (A @ np.vstack([np.cos(t), np.sin(t)])).T
    lo, hi = edge.min(0) - 0.35, edge.max(0) + 0.35
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160))
    f = (np.maximum(np.column_stack([gx.ravel(), gy.ravel()]) @ W.T + b, 0) @ a + c).reshape(gx.shape)
    ax.contourf(gx, gy, f, levels=[-np.inf, 0, np.inf], colors=[LABEL0, LABEL1], alpha=0.15)
    if f.min() < 0 < f.max():
        ax.contour(gx, gy, f, levels=[0], colors="k", linewidths=2.4)
    ax.plot([], [], color="k", lw=2.4, label="net decision boundary")
    ax.scatter(x[:, 0], x[:, 1], s=4, lw=0, c=np.where(labels == 1, LABEL1, LABEL0), alpha=0.8)
    ax.plot(edge[:, 0], edge[:, 1], color="k", lw=0.8)
    for p0, n in boundaries:                              # task boundary: the line through p0 normal to n
        d = np.array([-n[1], n[0]])
        ax.plot(*np.stack([p0 - 3 * d, p0 + 3 * d]).T, color="k", ls="--", lw=1.6)
    for h, w in enumerate(W):
        nw = np.linalg.norm(w)
        if nw < 1e-9:
            continue
        u = w / nw
        d, p0 = np.array([-u[1], u[0]]), -b[h] / nw * u      # w·x + b = 0 is the line through p0 along d
        ax.plot(*np.stack([p0 - 6 * d, p0 + 6 * d]).T, color=colors[h], lw=1.6, ls=":" if silent[h] else "-",
                label=unit_label(h, a[h], b[h], act[h]))
        ax.annotate("", xy=p0 + 0.45 * u, xytext=p0, arrowprops=dict(arrowstyle="->", color=colors[h], lw=1.6))
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("input $x_1$")
    ax.set_ylabel("input $x_2$")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85, handlelength=1.5)


def draw_weights_2d(ax, W_hist, i, silent, colors, dead_rows, lim):
    for h in range(W_hist.shape[1]):
        ax.plot(*W_hist[: i + 1, h].T, color=colors[h], lw=1.0, alpha=0.6)
        ax.plot(*W_hist[i, h], marker="o", ms=8, color=colors[h], mfc="white" if silent[h] else colors[h], mew=2)
        if h in dead_rows:
            ax.plot(*W_hist[dead_rows[h], h], marker="X", ms=10, color="k", mec="white")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("weight $w_1$")
    ax.set_ylabel("weight $w_2$")


# ── D = 3 ──────────────────────────────────────────────────────────────────────

def plane_patch(ax, n, p0, half, **kw):
    """The square of half-width `half` around p0 in the plane through p0 normal to n."""
    n = n / np.linalg.norm(n)
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    s, t = np.meshgrid([-half, half], [-half, half])
    P = p0[:, None, None] + e1[:, None, None] * s + e2[:, None, None] * t
    ax.plot_surface(P[0], P[1], P[2], shade=False, **kw)


def draw_input_3d(ax, x, labels, A, boundaries, W, b, a, silent, act, colors, azim):
    ax.scatter(*x.T, s=3, lw=0, c=np.where(labels == 1, LABEL1, LABEL0), alpha=0.6, depthshade=False)
    for p0, n in boundaries:
        plane_patch(ax, n, p0, 1.1, color="0.35", alpha=0.18, edgecolor="k", lw=0.8)
    half = np.diag(A) + 0.3
    for h, w in enumerate(W):
        nw = np.linalg.norm(w)
        if nw < 1e-9:
            continue
        p0 = -b[h] / nw ** 2 * w                       # the point of w·x + b = 0 closest to the centre
        plane_patch(ax, w, p0, 0.9, color=colors[h], alpha=0.05 if silent[h] else 0.18, edgecolor=colors[h],
                    lw=0.6 if silent[h] else 1.2, linestyle=":" if silent[h] else "-")
        ax.quiver(*p0, *(0.5 * w / nw), color=colors[h], lw=1.8, arrow_length_ratio=0.3)
        ax.plot([], [], color=colors[h], ls=":" if silent[h] else "-", lw=1.6, label=unit_label(h, a[h], b[h], act[h]))
    for setter, hv in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), half):
        setter(-hv, hv)
    ax.set_box_aspect(2 * half)
    ax.set_xlabel("input $x_1$")
    ax.set_ylabel("input $x_2$")
    ax.set_zlabel("input $x_3$")
    ax.view_init(elev=18, azim=azim)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85, handlelength=1.5)


def draw_weights_3d(ax, W_hist, i, silent, colors, dead_rows, lim, azim):
    for h in range(W_hist.shape[1]):
        ax.plot(*W_hist[: i + 1, h].T, color=colors[h], lw=1.0, alpha=0.6)
        ax.scatter(*W_hist[i, h], s=55, color="white" if silent[h] else colors[h], edgecolor=colors[h], lw=2,
                   depthshade=False)
        if h in dead_rows:
            ax.scatter(*W_hist[dead_rows[h], h], marker="X", s=80, color="k", depthshade=False)
    for e in np.eye(3):
        ax.plot(*np.stack([-lim * e, lim * e]).T, color="k", lw=0.4)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("weight $w_1$")
    ax.set_ylabel("weight $w_2$")
    ax.set_zlabel("weight $w_3$")
    ax.view_init(elev=18, azim=azim)


# ── Accuracy panel and the GIF ─────────────────────────────────────────────────

def draw_accuracy(ax, arr, t, i, schedule, S, H, silent):
    steps = arr["step"]
    shade_tasks(ax, schedule, S)
    idx = np.flatnonzero(steps <= t)
    ax.plot(steps[idx], arr["test_acc"][idx, arr["task"][idx]], color=CONTINUED, lw=2.4, zorder=3,
            label="continued net (the units shown)")
    for q in range(len(schedule)):
        m = (arr["fresh_step"] <= t) & (arr["fresh_phase"] == q)
        if m.any():
            ax.plot(arr["fresh_step"][m], arr["fresh_test_acc"][m, schedule[q]], color=FRESH, lw=1.0, ls="--",
                    alpha=0.7, label="fresh net, new each phase (baseline)" if q == 0 else None)
    ax.axvline(t, color="k", lw=0.6)
    ax.set_xlim(0, len(schedule) * S)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out accuracy, current task")
    ax.legend(fontsize=7, loc="lower left")
    q = min(t // S, len(schedule) - 1)
    fq = np.flatnonzero((arr["fresh_phase"] == q) & (arr["fresh_step"] <= t))
    f_now = arr["fresh_test_acc"][fq[-1], schedule[q]] if len(fq) else np.nan
    task = int(arr["task"][i])
    ax.set_title(f"now: continued {arr['test_acc'][i, task]:.2f}, fresh {f_now:.2f}; "
                 f"live units {int((~silent).sum())} of {H}", fontsize=9, fontweight="bold")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=100, help="steps between frames right after a switch")
    p.add_argument("--window", type=int, default=1500, help="how long after a switch frames stay dense")
    p.add_argument("--coarse", type=int, default=500, help="steps between frames otherwise")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=70)
    p.add_argument("--spin", type=float, default=0.6, help="D = 3: degrees the view turns per frame")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    cfg, _, arr = load_seed(args.run_dir, args.seed)
    D, H = cfg.input_dim, cfg.width
    if D not in (2, 3):
        raise SystemExit("the animation needs a D = 2 or D = 3 run")
    tasks = build_seed_tasks(cfg)
    schedule, S, steps = cfg.schedule, cfg.steps_per_task, arr["step"]
    frames = frame_steps(len(schedule) * S, args.every, args.window, args.coarse, len(schedule), S)
    show = np.random.default_rng(0).permutation(cfg.n_train)[:600 if D == 2 else 500]
    x_show = tasks.inputs("train")[show]
    colors = unit_colors(H)
    lim = 1.1 * max(np.abs(arr["W"]).max(), 0.5)
    death = arr["death_step"]

    set_style()
    if D == 2:
        fig, (ax_in, ax_w, ax_acc) = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    else:
        fig = plt.figure(figsize=(16, 5.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.2])
        ax_in, ax_w = fig.add_subplot(gs[0], projection="3d"), fig.add_subplot(gs[1], projection="3d")
        ax_acc = fig.add_subplot(gs[2])

    def draw(t, k):
        i = log_index(steps, t)
        task = int(arr["task"][i])
        silent = arr["act_frac_train"][i] == 0
        dead_rows = {h: log_index(steps, death[h]) for h in range(H) if 0 <= death[h] <= t}
        W, b, a, act = arr["W"][i], arr["b"][i], arr["a"][i], arr["act_frac_train"][i]
        for ax in (ax_in, ax_w, ax_acc):
            ax.cla()
        labels = tasks.labels_train[task, show]
        if D == 2:
            draw_input_2d(ax_in, x_show, labels, tasks.A, tasks.boundaries(task), W, b, a, float(arr["c"][i]),
                          silent, act, colors)
            draw_weights_2d(ax_w, arr["W"], i, silent, colors, dead_rows, lim)
        else:
            azim = -60 + args.spin * k
            draw_input_3d(ax_in, x_show, labels, tasks.A, tasks.boundaries(task), W, b, a, silent, act, colors, azim)
            draw_weights_3d(ax_w, arr["W"], i, silent, colors, dead_rows, lim, azim)
        ax_in.set_title(f"Input space: labels of task {task_name(task)}", fontsize=9)
        ax_w.set_title("Weight space (X: the unit died)", fontsize=9)
        draw_accuracy(ax_acc, arr, t, i, schedule, S, H, silent)
        q = min(t // S, len(schedule) - 1)
        fig.suptitle(f"seed {args.seed}, width {H}: step {t:,} — phase {q}, training task {task_name(schedule[q])}",
                     fontsize=12)
        if D == 2:
            fig.tight_layout(rect=(0, 0, 1, 0.93))
        else:
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.88, wspace=0.28)

    out = args.out or args.run_dir / "figures" / f"seed{args.seed}_units.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.set_dpi(args.dpi)
    images = []
    for k, t in enumerate(frames):
        draw(int(t), k)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy(), "RGBA"))
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / args.fps), loop=0)
    plt.close(fig)
    print(f"-> {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
