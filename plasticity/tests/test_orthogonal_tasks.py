"""The orthogonal-tasks toy: data and labels, config and run names, the net, a short run, summaries, the CLI."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from orthogonal_tasks.config import Config, run_name
from orthogonal_tasks.data import (build_tasks, resolve_input_dim, sample_unit_ball, task_normals,
                                   task_schedule)
from orthogonal_tasks.network import DTYPE, full_batch_grads, make_net, params, redraw_silent_units
from orthogonal_tasks.run import main
from orthogonal_tasks.summary import aggregate, load_seed, save_seed, summarize
from orthogonal_tasks.train import DeathWatch, run_seed

SHORT = dict(steps_per_task=60, phases=3, n_train=256, n_test=256, log_every=5)


# ── Data ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k, d", [(2, 2), (3, 3), (2, 5)])
def test_axis_normals_are_the_first_coordinate_axes(k, d):
    np.testing.assert_allclose(task_normals(k, d), np.eye(d)[:k])


def test_diagonal_normals_use_two_coordinates_and_c_uses_x3():
    v = task_normals(2, 3, "diagonal")
    np.testing.assert_allclose(v @ v.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(v[:, 2], 0)
    c = task_normals(3, 3, "diagonal-c")[2]
    np.testing.assert_allclose(c, np.array([1, 0, 1]) / np.sqrt(2))


def test_input_dimension_defaults_and_rejects_too_few_dims():
    assert resolve_input_dim(2) == 2 and resolve_input_dim(3) == 3
    with pytest.raises(ValueError):
        resolve_input_dim(3, 2)


def test_ball_samples_are_inside_and_fill_the_ball():
    u = sample_unit_ball(20_000, 2, np.random.default_rng(0))
    r = np.linalg.norm(u, axis=1)
    assert r.max() <= 1 and abs((r < 0.5).mean() - 0.25) < 0.02     # area fraction inside radius 1/2


def test_inputs_are_the_centred_ellipse_and_tasks_share_them():
    t = build_tasks(2, n_train=3000, n_test=10, seed=0)
    x = t.inputs("train")
    assert np.all((x[:, 0] / 1.0) ** 2 + (x[:, 1] / 0.5) ** 2 <= 1 + 1e-12)
    assert t.labels_train.shape == (2, 3000)


def test_halfspace_labels_are_the_axis_signs():
    t = build_tasks(2, n_train=500, n_test=10, seed=1)
    x = t.inputs("train")
    np.testing.assert_array_equal(t.labels_train, (x > 0).T.astype(int))


def test_stripe_labels_are_balanced_bands_no_line_can_separate():
    t = build_tasks(1, 2, n_train=4000, n_test=10, seed=0, label_rule="stripe")
    x, y = t.inputs("train"), t.labels_train[0]
    assert abs(y.mean() - 0.5) < 0.01
    np.testing.assert_array_equal(y, (np.abs(x[:, 0]) < t.half_widths[0]).astype(int))
    best = 0.0
    for th in np.linspace(0, np.pi, 90, endpoint=False):     # best single half-plane classifier
        proj = x @ np.array([np.cos(th), np.sin(th)])
        for c in np.quantile(proj, np.linspace(0.02, 0.98, 49)):
            acc = ((proj > c) == y).mean()
            best = max(best, acc, 1 - acc)
    assert best < 0.8


def test_xor_labels_are_balanced_uncorrelated_and_match_their_boundaries():
    t = build_tasks(2, n_train=4000, n_test=10, seed=0, label_rule="xor")
    u, x = t.u_train, t.inputs("train")
    np.testing.assert_array_equal(t.labels_train[0], (u[:, 0] * u[:, 1] > 0).astype(int))
    np.testing.assert_array_equal(t.labels_train[1], (np.abs(u[:, 1]) > np.abs(u[:, 0])).astype(int))
    assert abs(t.labels_train.mean(1) - 0.5).max() < 0.03
    assert abs(np.corrcoef(t.labels_train)[0, 1]) < 0.05
    for task in (0, 1):                                       # label = 1 where both boundary sides agree
        (p0, n0), (p1, n1) = t.boundaries(task)
        np.testing.assert_array_equal(t.labels_train[task], (((x - p0) @ n0) * ((x - p1) @ n1) > 0).astype(int))


def test_schedule_cycles_through_the_tasks():
    assert task_schedule(2, 5) == [0, 1, 0, 1, 0]
    assert Config(n_tasks=3, phases=4).schedule == [0, 1, 2, 0]


# ── Config and run names ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [dict(label_rule="xor", n_tasks=3), dict(task_normals="diagonal-c"),
                                 dict(bias_init="normal"), dict(scales=[1.0]), dict(width=0)])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        Config(**bad).resolved()


def test_run_names_are_readable():
    assert run_name(Config()) == "2d_AB-axes_thin-x2_4-units_10k-steps_10-switches"
    assert run_name(Config(label_rule="stripe", steps_per_task=20_000)).startswith("2d_AB-axes-stripes_thin-x2_")
    assert run_name(Config(label_rule="xor")).startswith("2d_AB-axes-xor_")
    assert run_name(Config(input_dim=3, task_normals="diagonal", scales=[0.5, 1, 1], steps_per_task=5000)) == \
        "3d_AB-diagonal_thin-x1_4-units_5k-steps_10-switches"
    assert run_name(Config(bias_init="zero", lr=0.01)).endswith("_zero-bias-init_lr0.01")


# ── Network ───────────────────────────────────────────────────────────────────

def test_uniform_bias_init_is_random_and_bounded_and_zero_init_is_zero():
    X = torch.tensor(build_tasks(2, n_train=256, n_test=10, seed=0).inputs("train"), dtype=DTYPE)
    b = make_net(Config(width=64).resolved(), 0, X).fc1.bias.detach().numpy()
    assert np.all(np.abs(b) <= 1 / np.sqrt(2)) and b.std() > 0.1
    b0 = make_net(Config(width=8, bias_init="zero").resolved(), 0, X).fc1.bias.detach().numpy()
    np.testing.assert_array_equal(b0, 0)


def test_every_unit_starts_alive_and_only_silent_units_are_redrawn():
    X = torch.tensor(build_tasks(2, n_train=256, n_test=10, seed=0).inputs("train"), dtype=DTYPE)
    net = make_net(Config(width=16).resolved(), 3, X)
    W, b, _, _ = params(net)
    assert ((X @ W.T + b).max(0).values > 0).all()
    with torch.no_grad():                                   # silence unit 1 everywhere: zero weights, b < 0
        net.fc1.weight[1] = 0.0
        net.fc1.bias[1] = -0.1
    keep = net.fc1.weight.detach().clone()
    assert redraw_silent_units(net, X, 3) >= 1
    W = net.fc1.weight.detach()
    others = [h for h in range(16) if h != 1]
    torch.testing.assert_close(W[others], keep[others])
    assert (X @ W[1] + net.fc1.bias.detach()[1]).max() > 0


def test_full_batch_gradient_matches_autograd_and_is_zero_for_a_silent_unit():
    torch.manual_seed(0)
    X = torch.randn(64, 2, dtype=DTYPE)
    Y = (torch.rand(2, 64) > 0.5).to(DTYPE)
    W = torch.randn(3, 2, dtype=DTYPE, requires_grad=True)
    b = torch.tensor([0.1, -0.2, -100.0], dtype=DTYPE, requires_grad=True)   # unit 2 is silent
    a = torch.randn(3, dtype=DTYPE, requires_grad=True)
    c = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
    gW, gb, ga, gc, loss, _ = full_batch_grads(W.detach(), b.detach(), a.detach(), c.detach(), X, Y)
    for k in range(2):
        f = torch.relu(X @ W.T + b) @ a + c
        L = F.binary_cross_entropy_with_logits(f, Y[k])
        dW, db, da, dc = torch.autograd.grad(L, (W, b, a, c))
        torch.testing.assert_close(gW[k], dW)
        torch.testing.assert_close(gb[k], db)
        torch.testing.assert_close(ga[k], da)
        torch.testing.assert_close(gc[k], dc)
        torch.testing.assert_close(loss[k], L)
    assert gW[:, 2].abs().max() == 0 and gb[:, 2].abs().max() == 0


# ── Training ──────────────────────────────────────────────────────────────────

def test_short_run_is_reproducible_and_logs_every_switch():
    cfg = Config(**SHORT, seed=0)
    a1, a2 = run_seed(cfg), run_seed(cfg)
    for k in ("W", "b", "test_acc", "fresh_W", "death_step"):
        np.testing.assert_array_equal(a1[k], a2[k])
    S, P = SHORT["steps_per_task"], SHORT["phases"]
    assert set(range(0, (P + 1) * S, S)) <= set(a1["step"].tolist())
    assert a1["step"][-1] == P * S and a1["task"].tolist()[:1] == [0]
    for p in range(P):
        fs = a1["fresh_step"][a1["fresh_phase"] == p]
        assert fs[0] == p * S and fs[-1] == (p + 1) * S
    assert a1["fresh_death_step"].shape == (P, cfg.width)


def test_death_watch_records_the_first_silent_step_only():
    X = torch.tensor(build_tasks(2, n_train=128, n_test=10, seed=0).inputs("train"), dtype=DTYPE)
    net = make_net(Config(width=3).resolved(), 0, X)
    watch = DeathWatch(3)
    watch.check(net, X, 1)
    assert (watch.step == -1).all()
    with torch.no_grad():
        net.fc1.bias[2] = -100.0
    watch.check(net, X, 7)
    watch.check(net, X, 9)
    assert watch.step.tolist() == [-1, -1, 7]


def test_logged_activity_agrees_with_the_death_steps():
    arr = run_seed(Config(**{**SHORT, "steps_per_task": 400, "phases": 4}, seed=2, lr=0.5))
    for h, s in enumerate(arr["death_step"]):
        after = arr["step"] >= s
        if s >= 0:
            assert (arr["act_frac_train"][after, h] == 0).all()
        else:
            assert (arr["act_frac_train"][:, h] > 0).all()


# ── Summaries, saving, CLI ────────────────────────────────────────────────────

def test_summary_gaps_deaths_and_round_trip(tmp_path):
    cfg = Config(**SHORT, seed=1).resolved()
    arr = run_seed(cfg)
    s = summarize(cfg, arr)
    assert len(s["phases"]) == SHORT["phases"] and len(s["units"]) == cfg.width
    for ph in s["phases"]:
        assert ph["fresh_minus_continued_acc"] == pytest.approx(ph["fresh_final_test_acc"]
                                                                - ph["continued_final_test_acc"])
    assert s["died"] == sum(u["death"] is not None for u in s["units"])
    save_seed(tmp_path, cfg, arr, s)
    cfg2, s2, arr2 = load_seed(tmp_path, 1)
    assert cfg2 == cfg and s2 == json.loads(json.dumps(s))
    np.testing.assert_array_equal(arr2["W"], arr["W"])
    agg = aggregate(tmp_path)
    assert agg["seeds"] == [1] and len(agg["gap_mean_per_phase"]) == SHORT["phases"]


def test_cli_writes_a_run_folder(tmp_path, capsys):
    run_dir = main(["--steps-per-task", "40", "--phases", "2", "--n-train", "128", "--n-test", "64",
                    "--seeds", "0", "1", "--out", str(tmp_path)])
    assert run_dir.name == "2d_AB-axes_thin-x2_4-units_40-steps_1-switches_128-train"
    for s in (0, 1):
        assert (run_dir / f"seed{s}" / "arrays.npz").exists()
    assert (run_dir / "aggregate.json").exists()
    assert "All seeds [0, 1]" in capsys.readouterr().out
