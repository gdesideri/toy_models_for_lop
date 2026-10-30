"""
animate_orthogonal_tasks.py — GIF of the hidden units of one seed of experiments/run_orthogonal_tasks.py
(D = 2 or D = 3 runs: the weight space is then drawn exactly).

Each frame, one row per condition (shifted, centered):
    input space    the train inputs coloured by the label of the task being trained, the net's own decision
                   boundary (D = 2: solid black line where the logit is 0, background shaded by predicted class),
                   the task boundary
                   (black dashed; both edges of the band for --label-rule stripe, both lines of the
                   XOR for xor), and every unit's own boundary w·x + b = 0 (a line through the
                   origin when there is no hidden bias) with an arrow on its active side; dotted =
                   the unit is silent on the whole train sample
    weight space   each unit's weight vector w_k with its trail so far, and an X where a unit lost its training signal
    accuracy       held-out accuracy on the current task so far: continued net vs fresh nets
In D = 3 both spaces are 3-D views that turn slowly: the task boundary is a grey plane, each unit's boundary
w·x + b = 0 a plane in its colour (faint and dotted when silent) with an arrow towards its active side.
The inactive cone is not drawn (with a trainable bias units also die outside it).

Usage (from plasticity/)
    uv run python -m visualization_scripts.toy_models.animate_orthogonal_tasks \
        --run-dir results/orthogonal_tasks/<run> --seed 0 [--every 100] [--window 1500] [--coarse 500] [--fps 12] [--out <file.gif>]
        [--end-window 1500 --slow 2.5 --hold 1.5]   optional: play the last steps of each phase slower and
                                                    hold on the last state before each switch
--show-error (D = 2): line width in input space and dot size in weight space grow with the error each unit
carries, |∂L/∂b_k| = |a_k · mean over the train set of (p − y)·1[unit k active]|, normalised to [0, 1].
Default output: <run>/figures/seed<s>_units.gif (with --cond shifted: seed<s>_units_shifted.gif, one row only)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.run_orthogonal_tasks import seed_for  # noqa: E402
from src import unit_geometry as geo  # noqa: E402
from tasks.orthogonal_halfspaces import build_tasks  # noqa: E402
from visualization_scripts.common import set_style  # noqa: E402
from visualization_scripts.toy_models.plot_orthogonal_tasks import (  # noqa: E402
    COND_TITLE, CONTINUED, FRESH, TASK_TINTS, death_steps, fmt_vec, load_seed,
    log_index, task_name, unit_colors)

LABEL0, LABEL1 = "#9ecae1", "#E69F00"


def frame_steps(total: int, every: int, window: int, coarse: int, end_window: int, schedule_len: int, S: int,
                logged: np.ndarray) -> np.ndarray:
    """Every `every` steps during the first `window` steps after each switch (where units move) and the last
    `end_window` steps before it (where they settle), every `coarse` steps otherwise, plus every switch and the
    last logged step before each switch."""
    fine = [np.arange(q * S, min(q * S + window, total) + 1, every) for q in range(schedule_len)]
    ends = [np.arange(max((q + 1) * S - end_window, 0), (q + 1) * S, every) for q in range(schedule_len)]
    last = [logged[np.searchsorted(logged, (q + 1) * S) - 1] for q in range(schedule_len)] if end_window else []
    return np.unique(np.concatenate(fine + ends + [np.arange(0, total + 1, coarse), np.arange(schedule_len + 1) * S,
                                                   np.array(last, dtype=int)]))


def frame_durations(steps: np.ndarray, S: int, end_window: int, fps: int, slow: float, hold: float,
                    logged: np.ndarray) -> list[int]:
    """Milliseconds per frame: 1/fps normally, `slow` times longer in the last `end_window` steps of a phase,
    and `hold` seconds on the last logged state before each switch."""
    base = 1000 / fps
    last = {int(logged[np.searchsorted(logged, (q + 1) * S) - 1]) for q in range(int(steps.max()) // S + 1)}
    out = []
    for t in steps:
        to_switch = (-int(t)) % S
        d = base * slow if 0 < to_switch <= end_window else base
        out.append(int(round(hold * 1000 if hold > 0 and int(t) in last else d)))
    return out


def draw_input(ax, x, labels, mu, A, v, h_task, W, b, silent, colors, act, a, xor_dirs=None, err=None, c=None):
    t = np.linspace(0, 2 * np.pi, 200)
    edge = mu + (A @ np.vstack([np.cos(t), np.sin(t)])).T
    lo = np.minimum(edge.min(0), 0) - 0.35
    hi = np.maximum(edge.max(0), 0) + 0.35
    if c is not None:       # the net's decision: shaded by predicted class, solid black line where the logit is 0
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160))
        G = np.column_stack([gx.ravel(), gy.ravel()])
        f = (np.maximum(G @ W.T + b, 0) @ a + c).reshape(gx.shape)
        ax.contourf(gx, gy, f, levels=[-np.inf, 0, np.inf], colors=[LABEL0, LABEL1], alpha=0.15)
        if f.min() < 0 < f.max():
            ax.contour(gx, gy, f, levels=[0], colors="k", linewidths=2.4)
        ax.plot([], [], color="k", lw=2.4, label="net decision boundary")
    ax.scatter(x[:, 0], x[:, 1], s=4, lw=0, c=np.where(labels == 1, LABEL1, LABEL0), alpha=0.8)
    ax.plot(edge[:, 0], edge[:, 1], color="k", lw=0.8)
    dv = np.array([-v[1], v[0]])        # task boundary v·(x − μ) = 0, or a stripe's edges v·(x − μ) = ±h
    lines = ([(mu + off * v, dv) for off in ([0.0] if h_task is None else [-h_task, h_task])]
             if xor_dirs is None else [(mu, d) for d in xor_dirs])      # XOR: its two boundary lines through μ
    for c, dv in lines:
        ax.plot([c[0] - 3 * dv[0], c[0] + 3 * dv[0]], [c[1] - 3 * dv[1], c[1] + 3 * dv[1]], color="k", ls="--",
                lw=1.6)
    ax.plot(0, 0, marker="P", color="#D55E00", ms=7)
    for h, w in enumerate(W):
        n = np.linalg.norm(w)
        if n < 1e-9:
            continue
        u = w / n
        d = np.array([-u[1], u[0]])
        p0 = -b[h] / n * u          # the boundary w·x + b = 0 is the line through p0 along d
        ls = ":" if silent[h] else "-"
        lw = 1.6 if err is None else 0.8 + 6 * err[h]
        txt = "" if err is None else f", err {err[h]:.2f}"
        ax.plot([p0[0] - 6 * d[0], p0[0] + 6 * d[0]], [p0[1] - 6 * d[1], p0[1] + 6 * d[1]], color=colors[h],
                lw=lw, ls=ls, label=f"unit {h}: a={a[h]:+.1f}, b={b[h]:+.1f}, active {act[h]:.0%}{txt}")
        ax.annotate("", xy=p0 + 0.45 * u, xytext=p0, arrowprops=dict(arrowstyle="->", color=colors[h], lw=lw))
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("input $x_1$")
    ax.set_ylabel("input $x_2$")
    ax.legend(fontsize=7.5, loc="lower right" if mu[1] > 0 else "upper left", framealpha=0.85, handlelength=1.5)


def draw_weights_2d(ax_w, arr, i, steps, t, silent, colors, deaths, lim, H, err=None):
    for h in range(H):
        ax_w.plot(arr["W"][: i + 1, h, 0], arr["W"][: i + 1, h, 1], color=colors[h], lw=1.0, alpha=0.6)
        ms = 8 if err is None else 5 + 16 * err[h]
        ax_w.plot(*arr["W"][i, h], marker="o", ms=ms, color=colors[h], mfc="white" if silent[h] else colors[h],
                  mew=2, zorder=3)
        if h in deaths and deaths[h]["step"] <= t:
            j = log_index(steps, deaths[h]["step"])
            ax_w.plot(*arr["W"][j, h], marker="X", ms=10, color="k", mec="white")
    ax_w.axhline(0, color="k", lw=0.4)
    ax_w.axvline(0, color="k", lw=0.4)
    ax_w.set_xlim(-lim, lim)
    ax_w.set_ylim(-lim, lim)
    ax_w.set_aspect("equal")
    ax_w.set_xlabel("weight $w_1$")
    ax_w.set_ylabel("weight $w_2$")


def plane_basis(n):
    """Two orthonormal vectors spanning the plane orthogonal to n (D = 3)."""
    n = n / np.linalg.norm(n)
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def plane_patch(ax, n, c, half, **kw):
    """The square of half-width `half` centred at c in the plane through c orthogonal to n."""
    e1, e2 = plane_basis(n)
    g = np.linspace(-half, half, 2)
    s, t = np.meshgrid(g, g)
    P = c[:, None, None] + e1[:, None, None] * s + e2[:, None, None] * t
    ax.plot_surface(P[0], P[1], P[2], shade=False, **kw)


def draw_input_3d(ax, x, labels, mu, A, v, h_task, W, b, silent, colors, act, a, azim):
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=3, lw=0, c=np.where(labels == 1, LABEL1, LABEL0), alpha=0.6,
               depthshade=False)
    for off in ([0.0] if h_task is None else [-h_task, h_task]):
        plane_patch(ax, v, mu + off * v, 1.1, color="0.35", alpha=0.18, edgecolor="k", lw=0.8)
    ax.scatter([0], [0], [0], marker="P", color="#D55E00", s=50)
    lo = np.minimum(mu - np.diag(A), 0) - 0.3
    hi = np.maximum(mu + np.diag(A), 0) + 0.3
    for h, w in enumerate(W):
        n = np.linalg.norm(w)
        if n < 1e-9:
            continue
        c = mu - (w @ mu + b[h]) / n ** 2 * w             # the point of w·x + b = 0 closest to the cloud centre
        plane_patch(ax, w, c, 0.9, color=colors[h], alpha=0.05 if silent[h] else 0.18, edgecolor=colors[h],
                    lw=0.6 if silent[h] else 1.2, linestyle=":" if silent[h] else "-")
        ax.quiver(*c, *(0.5 * w / n), color=colors[h], lw=1.8, arrow_length_ratio=0.3)
        ax.plot([], [], color=colors[h], ls=":" if silent[h] else "-", lw=1.6,
                label=f"unit {h}: a={a[h]:+.1f}, b={b[h]:+.1f}, active {act[h]:.0%}")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_xlabel("input $x_1$")
    ax.set_ylabel("input $x_2$")
    ax.set_zlabel("input $x_3$")
    ax.view_init(elev=18, azim=azim)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85, handlelength=1.5)


def draw_weights_3d(ax, W_hist, i, steps, silent, colors, dead_at, lim, azim):
    for h in range(W_hist.shape[1]):
        ax.plot(*W_hist[: i + 1, h].T, color=colors[h], lw=1.0, alpha=0.6)
        ax.scatter(*W_hist[i, h], s=55, color="white" if silent[h] else colors[h], edgecolor=colors[h], lw=2,
                   depthshade=False)
        if h in dead_at:
            ax.scatter(*W_hist[log_index(steps, dead_at[h]), h], marker="X", s=80, color="k", depthshade=False)
    for e in np.eye(3):
        ax.plot(*np.stack([-lim * e, lim * e]).T, color="k", lw=0.4)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("weight $w_1$")
    ax.set_ylabel("weight $w_2$")
    ax.set_zlabel("weight $w_3$")
    ax.view_init(elev=18, azim=azim)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", "--run_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=100, help="steps between frames right after a switch")
    p.add_argument("--window", type=int, default=1500, help="how long after a switch frames stay dense")
    p.add_argument("--coarse", type=int, default=500, help="steps between frames otherwise")
    p.add_argument("--end-window", "--end_window", type=int, default=0,
                   help="how long before a switch frames are dense and slowed down")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--slow", type=float, default=1.0, help="slow-down factor in the last --end-window steps of a phase")
    p.add_argument("--hold", type=float, default=0.0, help="seconds held on the last state before each switch (0 = no hold)")
    p.add_argument("--dpi", type=int, default=70)
    p.add_argument("--spin", type=float, default=0.6, help="D = 3: degrees the view turns per frame")
    p.add_argument("--cond", choices=["shifted", "centered"], default=None,
                   help="draw only this condition (one row); output name gets a _<cond> suffix")
    p.add_argument("--show-error", "--show_error", action="store_true",
                   help="D = 2: unit line width and weight-dot size grow with the error the unit carries, |∂L/∂b_k| "
                        "on the current task divided by its 99th percentile over the run (clipped to [0, 1])")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    cfg, summary, arrays = load_seed(args.run_dir, args.seed)
    D = cfg["input_dim"]
    if D not in (2, 3):
        raise SystemExit("the animation needs a D = 2 or D = 3 run (the weight space is exact only there)")
    set_style()
    conds = list(arrays) if args.cond is None else [args.cond]
    tasks = build_tasks(cfg["n_tasks"], D, cfg["n_train"], cfg["n_test"], seed_for(args.seed, "data"),
                        mu=cfg["mu"], scales=cfg["scales"], normals=cfg.get("task_normals", "axes"),
                        label_rule=cfg.get("label_rule", "halfspace"))
    show = np.random.default_rng(0).permutation(cfg["n_train"])[:600 if D == 2 else 500]
    first = arrays[conds[0]]
    schedule, S = first["schedule"].tolist(), int(first["steps_per_task"])
    T = len(schedule) * S
    logged = first["step"]
    steps_all = frame_steps(T, args.every, args.window, args.coarse, args.end_window, len(schedule), S, logged)
    durations = frame_durations(steps_all, S, args.end_window, args.fps, args.slow, args.hold, logged)
    H = cfg["width"]
    colors = unit_colors(H)
    lims = {c: 1.1 * max(np.abs(arrays[c]["W"]).max(), 0.5) for c in conds}
    deaths = {c: death_steps(summary["conditions"][c]) for c in conds}
    err_norm = {}
    if args.show_error:     # |∂L/∂b_k| on the current task, / its 99th percentile over the run, clipped to [0, 1]
        for c in conds:
            g = np.abs(arrays[c]["grad_b"][np.arange(len(arrays[c]["step"])), arrays[c]["task"]])
            err_norm[c] = np.clip(g / max(np.percentile(g, 99), 1e-12), 0, 1)

    if D == 2:
        fig, axes = plt.subplots(len(conds), 3, figsize=(14, 4.6 * len(conds)), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    else:
        fig = plt.figure(figsize=(16, 5.4 * len(conds)))
        gs = fig.add_gridspec(len(conds), 3, width_ratios=[1.2, 1, 1.2])
        axes = np.array([[fig.add_subplot(gs[r, 0], projection="3d"), fig.add_subplot(gs[r, 1], projection="3d"),
                          fig.add_subplot(gs[r, 2])] for r in range(len(conds))])

    def draw(t):
        azim = -60 + args.spin * int(np.searchsorted(steps_all, t))
        for r, cond in enumerate(conds):
            arr = arrays[cond]
            steps = arr["step"]
            i = log_index(steps, t)
            task = int(arr["task"][i])
            silent = (arr["act_frac_train"][i] == 0) | (arr["status"][i] != geo.ACTIVE)
            ax_in, ax_w, ax_acc = axes[r]
            for ax in axes[r]:
                ax.cla()
            draw_in = draw_input if D == 2 else draw_input_3d
            extra = () if D == 2 else (azim,)
            h_task = None if tasks.half_widths is None else tasks.half_widths[task]
            if tasks.pairs is not None and D == 2:   # XOR: boundaries p·u = 0, q·u = 0 are the lines along A q, A p
                dirs = [arr["A"] @ tasks.pairs[task, 1], arr["A"] @ tasks.pairs[task, 0]]
                extra = ([d / np.linalg.norm(d) for d in dirs],)
            err = err_norm[cond][i] if cond in err_norm else None
            kw = {"err": err, "c": float(arr["c"][i])} if D == 2 else {}
            draw_in(ax_in, tasks.inputs("train", cond)[show], tasks.labels_train[task, show], arr["mu"], arr["A"],
                    arr["normals"][task], h_task, arr["W"][i], arr["b"][i], silent, colors,
                    arr["act_frac_train"][i], arr["a"][i], *extra, **kw)
            ax_in.set_title(f"Input space, {COND_TITLE[cond]}: labels of task {task_name(task)}", fontsize=9)

            lim = lims[cond]
            if D == 3:
                dead_at = {h: d["step"] for h, d in deaths[cond].items() if d["step"] <= t}
                draw_weights_3d(ax_w, arr["W"], i, steps, silent, colors, dead_at, lim, azim)
                ax_w.set_title(f"Weight space, μ = {fmt_vec(arr['mu'])}", fontsize=9)
            else:
                draw_weights_2d(ax_w, arr, i, steps, t, silent, colors, deaths[cond], lim, H, err)
                ax_w.set_title(f"Weight space, μ = {fmt_vec(arr['mu'])}"
                               + ("; size = error carried |∂L/∂b|" if err is not None else ""), fontsize=9)


            for q, tq in enumerate(schedule):
                ax_acc.axvspan(q * S, (q + 1) * S, color=TASK_TINTS[tq], alpha=0.08, lw=0)
            idx = np.flatnonzero(steps <= t)
            ax_acc.plot(steps[idx], arr["test_acc"][idx, arr["task"][idx]], color=CONTINUED, lw=2.4, zorder=3,
                        label="continued net (the units shown)")
            fm = arr["fresh_step"] <= t
            for q in range(len(schedule)):
                m = fm & (arr["fresh_phase"] == q)
                if m.any():
                    ax_acc.plot(arr["fresh_step"][m], arr["fresh_test_acc"][m, schedule[q]], color=FRESH, lw=1.0,
                                ls="--", alpha=0.7, label="fresh net, new each phase (baseline)" if q == 0 else None)
            ax_acc.axvline(t, color="k", lw=0.6)
            ax_acc.set_xlim(0, T)
            ax_acc.set_ylim(0.4, 1.02)
            ax_acc.set_xlabel("training step")
            ax_acc.set_ylabel("held-out accuracy, current task")
            ax_acc.legend(fontsize=7, loc="lower left")
            q_now = min(t // S, len(schedule) - 1)
            fq = np.flatnonzero((arr["fresh_phase"] == q_now) & (arr["fresh_step"] <= t))
            f_now = arr["fresh_test_acc"][fq[-1], schedule[q_now]] if len(fq) else np.nan
            ax_acc.set_title(f"now: continued {arr['test_acc'][i, task]:.2f}, fresh {f_now:.2f}; "
                             f"live units {int((~silent).sum())} of {H}", fontsize=9, fontweight="bold")
        q = min(t // S, len(schedule) - 1)
        fig.suptitle(f"seed {args.seed}, width {H}: step {t:,} — phase {q}, training task {task_name(schedule[q])}",
                     fontsize=12)
        if D == 2:
            fig.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.91, wspace=0.28, hspace=0.18)

    suffix = "" if args.cond is None else f"_{args.cond}"
    out = args.out or args.run_dir / "figures" / f"seed{args.seed}_units{suffix}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.set_dpi(args.dpi)
    images = []
    for t in steps_all:
        draw(t)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy(), "RGBA"))  # as matplotlib's PillowWriter
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0)
    plt.close(fig)
    print(f"-> {out} ({len(steps_all)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
