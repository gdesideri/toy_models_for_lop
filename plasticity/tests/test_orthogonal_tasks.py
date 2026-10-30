"""The orthogonal-tasks toy: task geometry, the inactivity criterion, angles, and a short run."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.run_orthogonal_tasks import (EVENT_INFO, EVENT_SIGN, EVENT_SILENT, EVENT_STATUS, Config, deaths,
                                              full_batch_grads, make_net, params, print_seed_report, run_condition,
                                              run_name, run_seed)
from src import unit_geometry as geo
from tasks.orthogonal_halfspaces import (build_tasks, default_mu, default_scales, resolve_input_dim,
                                         sample_unit_ball, task_labels, task_normals, task_schedule)

SHIFTED_MU, ELLIPSE = np.array([0.0, 1.0]), np.diag([1.0, 0.5])


# ── Task normals and dimensions ───────────────────────────────────────────────

@pytest.mark.parametrize("k, d", [(2, 2), (3, 3), (2, 5), (4, 7)])
def test_task_normals_are_orthonormal_coordinate_axes(k, d):
    v = task_normals(k, d)
    assert v.shape == (k, d)
    np.testing.assert_allclose(v @ v.T, np.eye(k))
    np.testing.assert_allclose(v, np.eye(d)[:k])


def test_default_dimension_is_max_of_two_and_k():
    assert resolve_input_dim(1) == 2
    assert resolve_input_dim(2) == 2
    assert resolve_input_dim(5) == 5
    assert resolve_input_dim(2, 6) == 6


@pytest.mark.parametrize("k, d", [(3, 2), (5, 4)])
def test_fewer_dimensions_than_tasks_is_rejected(k, d):
    with pytest.raises(ValueError):
        resolve_input_dim(k, d)
    with pytest.raises(ValueError):
        task_normals(k, d)
    with pytest.raises(ValueError):
        Config(n_tasks=k, input_dim=d).resolved()


def test_default_geometry_is_the_shifted_ellipse_in_two_dimensions():
    np.testing.assert_allclose(default_mu(2), SHIFTED_MU)
    np.testing.assert_allclose(default_scales(2), [1.0, 0.5])
    cfg = Config().resolved()
    assert cfg.input_dim == 2 and cfg.mu == [0.0, 1.0] and cfg.scales == [1.0, 0.5]


# ── Inputs and labels ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [2, 3, 6])
def test_ball_samples_are_inside_and_uniform(d):
    u = sample_unit_ball(20000, d, np.random.default_rng(0))
    r = np.linalg.norm(u, axis=1)
    assert r.max() < 1
    assert np.mean(r < 0.5) == pytest.approx(0.5 ** d, abs=0.01)   # uniform in volume


def test_inputs_lie_in_the_support_ellipsoid():
    t = build_tasks(3, 3, n_train=500, n_test=500, seed=1, scales=[1.0, 2.0, 0.5])
    for split in ("train", "test"):
        u = (t.inputs(split) - t.mu) @ np.linalg.inv(t.A).T
        assert np.linalg.norm(u, axis=1).max() <= 1 + 1e-12


def test_two_dimensional_labels_are_the_two_axis_halfplanes_through_mu():
    t = build_tasks(2, n_train=1000, n_test=1000, seed=0)
    x = t.inputs("train")
    np.testing.assert_array_equal(t.labels_train[0], (x[:, 0] > 0.0).astype(int))   # y = 1[x1 > mu1]
    np.testing.assert_array_equal(t.labels_train[1], (x[:, 1] > 1.0).astype(int))   # y = 1[x2 > mu2]


@pytest.mark.parametrize("k, d", [(2, 2), (3, 3), (2, 4)])
def test_labels_are_identical_in_the_shifted_and_centered_conditions(k, d):
    t = build_tasks(k, d, n_train=2000, n_test=800, seed=3)
    for split, stored in (("train", t.labels_train), ("test", t.labels_test)):
        x_shift, x_cent = t.inputs(split, "shifted"), t.inputs(split, "centered")
        np.testing.assert_allclose(x_shift - x_cent, np.broadcast_to(t.mu, x_shift.shape))   # same u
        np.testing.assert_array_equal(task_labels(x_shift, t.mu, t.normals), stored)
        np.testing.assert_array_equal(task_labels(x_cent, np.zeros(d), t.normals), stored)


def test_tasks_share_inputs_and_only_labels_differ():
    t = build_tasks(3, 3, n_train=3000, n_test=100, seed=0)
    y = t.labels_train
    assert y.shape == (3, 3000)
    for j in range(3):
        assert 0.45 < y[j].mean() < 0.55                     # each hyperplane passes through μ
        for i in range(j):
            assert 0.4 < (y[i] != y[j]).mean() < 0.6         # orthogonal tasks: independent labels
    np.testing.assert_array_equal(t.inputs("train"), build_tasks(3, 3, 3000, 100, seed=0).inputs("train"))


def test_diagonal_normals_are_orthonormal_and_use_two_coordinates():
    v = task_normals(2, 3, "diagonal")
    np.testing.assert_allclose(v @ v.T, np.eye(2), atol=1e-12)
    np.testing.assert_array_equal(v[:, 2], 0.0)
    assert np.all(v[:, :2] != 0)
    with pytest.raises(ValueError):
        task_normals(3, 3, "diagonal")


def test_diagonal_c_adds_a_task_on_x1_and_x3():
    v = task_normals(3, 3, "diagonal-c")
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)
    np.testing.assert_allclose(v[2], np.array([1.0, 0.0, 1.0]) / np.sqrt(2))
    np.testing.assert_allclose(v[2] @ v[:2].T, [0.5, 0.5])          # 60° to A and B
    with pytest.raises(ValueError):
        task_normals(2, 3, "diagonal-c")


def test_stripe_labels_are_balanced_bands_around_the_centre():
    t = build_tasks(2, n_train=2000, n_test=2000, seed=0, label_rule="stripe")
    z = t.offsets("train")
    np.testing.assert_array_equal(t.labels_train, (np.abs(z @ t.normals.T) < t.half_widths).T.astype(int))
    assert np.all(np.abs(t.labels_train.mean(1) - 0.5) < 0.01)
    assert np.all(np.abs(t.labels_test.mean(1) - 0.5) < 0.05)
    shifted, centred = (t.labels_train, build_tasks(2, n_train=2000, n_test=2000, seed=0, label_rule="stripe",
                                                     mu=[0, 0]).labels_train)
    np.testing.assert_array_equal(shifted, centred)


def test_a_stripe_is_not_linearly_separable():
    # the band's inside sits between two outside regions along v, so no single threshold on any w·x + b fits it
    t = build_tasks(1, 2, n_train=4000, n_test=10, seed=0, label_rule="stripe")
    x, y = t.inputs("train"), t.labels_train[0]
    best = 0.0
    for ang in np.linspace(0, np.pi, 181):
        p = x @ np.array([np.cos(ang), np.sin(ang)])
        order = np.argsort(p)
        cum = np.concatenate([[0], np.cumsum(y[order])])                 # positives at or below each cut
        k = np.arange(len(y) + 1)
        acc = np.maximum((k - cum + (y.sum() - cum)) / len(y), (cum + (len(y) - k - (y.sum() - cum))) / len(y))
        best = max(best, acc.max())
    assert best < 0.8


def test_schedule_cycles_through_all_tasks_in_order():
    assert task_schedule(2, 3) == [0, 1, 0, 1, 0, 1]
    assert task_schedule(3, 2) == [0, 1, 2, 0, 1, 2]
    assert Config(n_tasks=4, cycles=2).resolved().schedule == [0, 1, 2, 3, 0, 1, 2, 3]
    assert Config(n_tasks=2, phases=11).resolved().schedule == [0, 1] * 5 + [0]


# ── Angles ────────────────────────────────────────────────────────────────────

def test_signed_angle_keeps_opposite_directions_apart():
    w = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1]], float)
    np.testing.assert_allclose(geo.signed_angle_2d(w), [0, 90, 180, -90, 45, -135])
    assert geo.angle_between(w[4], w[5]) == pytest.approx(180.0)   # not 0, as abs(dot) would give
    assert geo.angle_between(w[0], w[1]) == pytest.approx(90.0)
    assert geo.angle_between(w[0], 3 * w[0]) == pytest.approx(0.0)


def test_near_zero_vectors_have_no_angle():
    w = np.array([[1e-6, 0.0], [0.0, 1.0]])
    assert np.isnan(geo.signed_angle_2d(w)[0]) and geo.signed_angle_2d(w)[1] == 90
    assert np.isnan(geo.angle_between(w[0], [1.0, 0.0]))
    assert np.isnan(geo.directional_margin(w, SHIFTED_MU, ELLIPSE)[0])
    r, t = geo.radial_tangential(w, np.ones_like(w))
    assert np.isnan(r[0]) and np.isnan(t[0])


def test_plane_angle_of_a_projection_and_its_degenerate_case():
    v0, v1 = np.eye(4)[0], np.eye(4)[2]
    w = np.array([[1.0, 5.0, 1.0, 0.0], [0.0, 3.0, 0.0, 2.0]])
    ang = geo.plane_angle(w, v0, v1)
    assert ang[0] == pytest.approx(45.0)
    assert np.isnan(ang[1])                                  # projection onto the plane is zero


def test_radial_and_tangential_parts_of_an_update():
    r, t = geo.radial_tangential(np.array([[2.0, 0.0]]), np.array([[0.5, -0.3]]))
    assert r[0] == pytest.approx(0.5) and t[0] == pytest.approx(0.3)


# ── The inactivity criterion ──────────────────────────────────────────────────

def test_criterion_on_the_shifted_ellipse():
    W = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(geo.support_max(W, 0.0, SHIFTED_MU, ELLIPSE), [-0.5, 1.0, 1.5, 1.0])
    np.testing.assert_array_equal(geo.certified_inactive(W, 0.0, SHIFTED_MU, ELLIPSE), [True, False, False, False])


def test_inactive_cone_of_the_shifted_ellipse_is_the_analytic_wedge():
    # w = (sin φ, −cos φ) is inactive iff tan φ <= sqrt(3)/2: a 2 × 40.89° wedge around −μ
    half = np.degrees(np.arctan(np.sqrt(0.75)))
    arcs = geo.inactive_arcs_2d(SHIFTED_MU, ELLIPSE)
    assert len(arcs) == 1
    lo, hi = arcs[0]
    assert lo == pytest.approx(-90 - half, abs=1e-5) and hi == pytest.approx(-90 + half, abs=1e-5)
    theta = np.radians(np.linspace(-180, 180, 3601))
    dead = geo.certified_inactive(np.stack([np.cos(theta), np.sin(theta)], 1), 0.0, SHIFTED_MU, ELLIPSE)
    inside = (np.degrees(theta) >= lo) & (np.degrees(theta) <= hi)
    np.testing.assert_array_equal(dead, inside)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_support_max_is_the_true_maximum_over_the_support(d):
    rng = np.random.default_rng(d)
    for _ in range(20):
        mu, A = rng.normal(size=d), np.diag(rng.uniform(0.2, 2.0, d))
        if rng.random() < 0.5:
            A = A @ np.linalg.qr(rng.normal(size=(d, d)))[0]      # also any non-diagonal A
        w, b = rng.normal(size=d), rng.normal()
        m = geo.support_max(w, b, mu, A)
        u_star = A.T @ w / np.linalg.norm(A.T @ w)                  # the maximiser on the sphere
        assert w @ (mu + A @ u_star) + b == pytest.approx(m)
        u = rng.normal(size=(5000, d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        values = (mu + u @ A.T) @ w + b
        assert values.max() <= m + 1e-12
        assert geo.support_min(w, b, mu, A) == pytest.approx(2 * (w @ mu + b) - m)
        assert values.min() >= geo.support_min(w, b, mu, A) - 1e-12


def test_centered_control_has_no_nonzero_inactive_direction():
    rng = np.random.default_rng(0)
    for d in (2, 3, 6):
        W = rng.normal(size=(5000, d))
        A = np.diag(default_scales(d))
        assert not geo.certified_inactive(W, 0.0, np.zeros(d), A).any()
        assert (geo.unit_status(W, 0.0, np.zeros(d), A) == geo.ACTIVE).all()
    assert geo.inactive_arcs_2d(np.zeros(2), ELLIPSE) == []
    # the zero vector is collapsed, not rotated into a cone
    assert geo.unit_status(np.zeros((1, 2)), 0.0, np.zeros(2), ELLIPSE)[0] == geo.COLLAPSED


def test_orthogonal_to_a_task_normal_is_not_inactive():
    t = build_tasks(2, n_train=2000, n_test=10, seed=0)
    w = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0]])     # orthogonal to v_A, or to v_B
    for cond in ("shifted", "centered"):
        mu = t.condition_mu(cond)
        assert geo.angle_between(w[0], t.normals[0]) == pytest.approx(90.0)
        status = geo.unit_status(w, 0.0, mu, t.A)
        assert status[0] == geo.ACTIVE and status[2] == geo.ACTIVE
        assert status[1] == (geo.INACTIVE if cond == "shifted" else geo.ACTIVE)


def test_generalized_criterion_with_bias_in_higher_dimensions():
    mu, A = np.array([0.0, 0.0, 1.0]), np.diag([1.0, 1.0, 0.5])
    w = np.array([0.0, 0.0, -1.0])
    assert geo.support_max(w, 0.0, mu, A) == pytest.approx(-0.5)
    assert not geo.certified_inactive(w, 0.6, mu, A)          # a positive bias revives it
    assert geo.certified_inactive([1.0, 0.0, 0.0], -1.0, mu, A)   # and a negative bias kills (1, 0, 0)
    assert geo.directional_margin([1.0, 0.0, 0.0], mu, A) == pytest.approx(1.0)


def test_certified_inactivity_differs_from_silence_on_a_finite_sample():
    t = build_tasks(2, n_train=50, n_test=10, seed=0)
    X = t.inputs("train")
    dead = np.array([[0.0, -1.0]])
    assert geo.certified_inactive(dead, 0.0, t.mu, t.A)[0] and geo.silent_on_sample(dead, 0.0, X)[0]
    # nearly inactive: positive only on a sliver of the support near (0, 0.5) that 50 samples miss
    rng = np.random.default_rng(0)
    almost = None
    for _ in range(2000):
        th = np.radians(rng.uniform(-140, -40))
        w = np.array([[np.cos(th), np.sin(th)]])
        if 0 < geo.support_max(w, 0.0, t.mu, t.A)[0] < 0.01 and geo.silent_on_sample(w, 0.0, X)[0]:
            almost = w
            break
    assert almost is not None
    assert not geo.certified_inactive(almost, 0.0, t.mu, t.A)[0]


def test_death_mechanisms_are_classified_from_the_state_at_the_event():
    A, I, C = geo.ACTIVE, geo.INACTIVE, geo.COLLAPSED
    nan = np.nan
    #           step unit kind          old new    norm  dir_m  b/|w|  angle  sup_max train_dir_max test_act
    rows = [((10, 0, EVENT_SILENT, 0, 1), (2.0, 0.01, 0.0, -48.5, 0.02, -0.001, 0.001)),   # silent, not certified
            ((12, 0, EVENT_STATUS, A, I), (2.0, -0.1, 0.0, -90.0, -0.2, -0.3, 0.0)),        # later: ignored
            ((20, 1, EVENT_SIGN, 1, -1), (1.0, 0.5, 0.0, 0.0, 0.5, 0.4, 0.5)),              # readout flip: not a death
            ((25, 1, EVENT_STATUS, A, I), (1.5, -0.1, 0.0, -90.0, -0.15, -0.3, 0.0)),       # inside the cone
            ((30, 2, EVENT_STATUS, A, C), (1e-4, nan, nan, nan, 1e-4, nan, 0.5)),           # collapsed
            ((40, 3, EVENT_SILENT, 0, 1), (1.0, 0.5, -0.6, 10.0, -0.1, 0.4, 0.0)),          # negative bias
            ((50, 4, EVENT_STATUS, A, I), (1.0, -0.2, 0.0, -90.0, -0.2, -0.2, 0.0))]        # silent at init
    events = np.array([r[0] for r in rows])
    info = np.array([r[1] for r in rows])
    assert info.shape[1] == len(EVENT_INFO)
    init_status = np.array([A, A, A, A, A])
    init_silent = np.array([False, False, False, False, True])
    got = {d["unit"]: d for d in deaths(events, info, init_status, init_silent)}
    assert set(got) == {0, 1, 2, 3}
    assert got[0]["mechanism"] == "rotation_to_cone_edge" and not got[0]["certified_inactive"] and got[0]["step"] == 10
    assert got[1]["mechanism"] == "rotation_into_cone" and got[1]["certified_inactive"] and got[1]["step"] == 25
    assert got[2]["mechanism"] == "norm_collapse"
    assert got[3]["mechanism"] == "bias_driven"


# ── Network and gradients ─────────────────────────────────────────────────────

def test_network_has_no_hidden_bias_and_a_trainable_output_bias():
    net = make_net(Config().resolved(), seed=0)
    assert net.fc1.bias is None and net.fc2.bias is not None and net.fc2.bias.requires_grad
    assert tuple(net.fc1.weight.shape) == (4, 2)
    trainable = make_net(Config(hidden_bias="trainable").resolved(), seed=0)
    assert trainable.fc1.bias is not None


def test_analytic_full_batch_gradient_matches_autograd_and_is_zero_for_a_dead_unit():
    cfg = Config(hidden_bias="trainable").resolved()
    t = build_tasks(2, n_train=300, n_test=10, seed=0)
    net = make_net(cfg, seed=1)
    with torch.no_grad():
        net.fc1.weight[0] = torch.tensor([0.0, -2.0], dtype=torch.float64)   # certified inactive
        net.fc1.bias[0] = 0.0
    X = torch.tensor(t.inputs("train"), dtype=torch.float64)
    Y = torch.tensor(t.labels_train, dtype=torch.float64)
    gW, gb, ga, gc, loss, _ = full_batch_grads(*params(net), X, Y)
    for k in range(2):
        net.zero_grad()
        F.binary_cross_entropy_with_logits(net(X), Y[k]).backward()
        torch.testing.assert_close(gW[k], net.fc1.weight.grad)
        torch.testing.assert_close(gb[k], net.fc1.bias.grad)
        torch.testing.assert_close(ga[k], net.fc2.weight.grad[0])
        torch.testing.assert_close(gc[k], net.fc2.bias.grad[0])
    assert torch.all(gW[:, 0] == 0) and torch.all(ga[:, 0] == 0) and torch.all(gb[:, 0] == 0)


# ── Short runs ────────────────────────────────────────────────────────────────

SMALL = dict(steps_per_task=60, cycles=2, n_train=256, n_test=256, log_every=20, batch_size=32, lr=0.2)


def test_short_run_is_reproducible():
    a = run_seed(Config(seed=4, **SMALL))
    b = run_seed(Config(seed=4, **SMALL))
    for cond in ("shifted", "centered"):
        for key, val in a["arrays"][cond].items():
            np.testing.assert_array_equal(val, b["arrays"][cond][key], err_msg=f"{cond}/{key}")
    assert a["summary"] == b["summary"]


def test_short_run_follows_the_schedule_and_logs_every_switch():
    cfg = Config(n_tasks=3, seed=0, **SMALL).resolved()
    assert cfg.input_dim == 3
    t = build_tasks(3, 3, cfg.n_train, cfg.n_test, seed=0)
    arr, summ = run_condition(cfg, t, "shifted")
    S = cfg.steps_per_task
    assert [ph["task"] for ph in summ["phases"]] == [0, 1, 2, 0, 1, 2]
    for p, task in enumerate(cfg.schedule):
        assert np.all(arr["task"][arr["phase"] == p] == task)
        assert p * S in arr["step"]                                    # every switch state is logged
        f = arr["fresh_phase"] == p
        assert arr["fresh_step"][f][0] == p * S and arr["fresh_step"][f][-1] == (p + 1) * S
    assert arr["step"][-1] == 6 * S
    assert arr["W"].shape[1:] == (cfg.width, 3)
    assert arr["grad_w_norm"].shape[1:] == (3, cfg.width)


def test_both_conditions_start_from_the_same_weights_and_centered_never_has_inactive_units():
    res = run_seed(Config(seed=2, **SMALL))
    sh, ce = res["arrays"]["shifted"], res["arrays"]["centered"]
    np.testing.assert_array_equal(sh["W"][0], ce["W"][0])
    np.testing.assert_array_equal(sh["fresh_W"][0], ce["fresh_W"][0])
    assert not np.any(ce["status"] == geo.INACTIVE)
    assert not np.any(ce["fresh_status"] == geo.INACTIVE)


def test_certified_inactive_units_are_frozen_and_silent():
    # whatever happens, a unit in the cone gets exactly zero gradient and stays put
    for seed in range(4):
        arr = run_seed(Config(seed=seed, conditions=["shifted"], **SMALL))["arrays"]["shifted"]
        dead = arr["status"] == geo.INACTIVE
        assert np.all(arr["act_frac_train"][dead] == 0) and np.all(arr["act_frac_test"][dead] == 0)
        assert np.all(arr["grad_w_norm"].transpose(0, 2, 1)[dead] == 0)
        assert np.all(arr["support_max"][dead] <= 0)
        for h in range(arr["W"].shape[1]):
            rows = np.flatnonzero(dead[:, h])
            if len(rows):
                np.testing.assert_array_equal(arr["W"][rows, h], np.broadcast_to(arr["W"][rows[0], h], (len(rows), 2)))


def test_higher_dimension_than_tasks_runs():
    res = run_seed(Config(n_tasks=2, input_dim=4, seed=0, conditions=["shifted"], **SMALL))
    arr = res["arrays"]["shifted"]
    assert arr["W"].shape[-1] == 4 and np.all(np.isnan(arr["angle_2d"]))
    assert run_name(Config(n_tasks=2, input_dim=4)).startswith("4d_AB-axes_shift-x4_thin-x4_")


def test_report_prints_a_collapsed_unit_without_a_direction(capsys):
    summary = run_seed(Config(seed=0, conditions=["centered"], **SMALL))["summary"]
    unit = summary["conditions"]["centered"]["units"][0]
    unit["death"] = {"step": 5, "phase": 0, "task": 0, "mechanism": "norm_collapse", "dir_margin": None,
                     "norm_ratio_to_init": 0.004, "test_act_frac": 0.49}
    unit["angle_2d_final"] = None
    print_seed_report(summary)
    assert "norm_collapse, dir margin n/a" in capsys.readouterr().out


# ── Live initialisation ───────────────────────────────────────────────────────

def test_live_init_starts_every_unit_active_in_both_conditions_and_for_fresh_nets():
    res = run_seed(Config(seed=0, init="live", **SMALL))
    sh, ce = res["arrays"]["shifted"], res["arrays"]["centered"]
    for arr in (sh, ce):
        assert np.all(arr["init_status"] == geo.ACTIVE) and not arr["init_silent_train"].any()
        assert np.all(arr["fresh_init_status"] == geo.ACTIVE) and not arr["fresh_init_silent_train"].any()
    np.testing.assert_array_equal(sh["W"][0], ce["W"][0])            # same start in both conditions
    np.testing.assert_array_equal(sh["fresh_W"][0], ce["fresh_W"][0])
    assert sh["init_redraws"] > 0                                    # seed 0 has silent units by default


def test_live_init_only_redraws_the_silent_units():
    cfg = Config(seed=0, **SMALL).resolved()
    t = build_tasks(2, 2, cfg.n_train, cfg.n_test, seed=0)
    X = torch.tensor(t.inputs("train", "shifted"), dtype=torch.float64)
    default = make_net(cfg, seed=7)
    live = make_net(Config(seed=0, init="live", **SMALL).resolved(), seed=7, live_inputs=X)
    W0, W1 = default.fc1.weight.detach(), live.fc1.weight.detach()
    silent = (X @ W0.T).max(0).values <= 0
    assert silent.any()
    torch.testing.assert_close(W1[~silent], W0[~silent])             # live units untouched
    assert torch.all((X @ W1.T).max(0).values > 0)                   # every unit now fires somewhere
    torch.testing.assert_close(live.fc2.weight, default.fc2.weight)  # readout unchanged
    assert live.init_redraws >= int(silent.sum())
    with pytest.raises(ValueError):
        make_net(Config(init="live").resolved(), seed=7)             # needs the inputs


def test_default_init_is_unchanged_by_the_live_option():
    a = run_seed(Config(seed=1, conditions=["shifted"], **SMALL))["arrays"]["shifted"]
    assert a["init_redraws"] == 0 and np.all(a["fresh_init_redraws"] == 0)
    assert run_name(Config()).endswith("_kaiming-init") and run_name(Config(init="live")).endswith("_live-init")


def test_run_name_marks_stripes():
    assert run_name(Config(label_rule="stripe")).startswith("2d_AB-axes-stripes_shift-x2_")


def test_run_name_is_readable():
    assert (run_name(Config(steps_per_task=5000, phases=11, init="live"))
            == "2d_AB-axes_shift-x2_thin-x2_no-bias_4-units_5k-steps_10-switches_live-init")
    assert (run_name(Config(n_tasks=3, input_dim=3, task_normals="diagonal-c", mu=[1, 0, 0], scales=[0.5, 1, 1],
                            hidden_bias="trainable", lr=0.01))
            == "3d_ABC-diagonal_shift-x1_thin-x1_trainable-bias_4-units_4k-steps_11-switches_kaiming-init_lr0.01")


def test_xor_labels_are_balanced_and_uncorrelated():
    t = build_tasks(2, n_train=4000, n_test=10, seed=0, label_rule="xor")
    u = t.u_train
    np.testing.assert_array_equal(t.labels_train[0], (u[:, 0] * u[:, 1] > 0).astype(int))
    np.testing.assert_array_equal(t.labels_train[1], (np.abs(u[:, 1]) > np.abs(u[:, 0])).astype(int))
    assert abs(t.labels_train.mean(1) - 0.5).max() < 0.03
    assert abs(np.corrcoef(t.labels_train)[0, 1]) < 0.05


def test_run_name_marks_xor():
    assert run_name(Config(label_rule="xor")).startswith("2d_AB-axes-xor_shift-x2_")


def test_uniform_bias_init_is_random_and_bounded():
    cfg = Config(hidden_bias="trainable", bias_init="uniform", width=64).resolved()
    b = make_net(cfg, 0).fc1.bias.detach().numpy()
    assert np.all(np.abs(b) <= 1 / np.sqrt(cfg.input_dim)) and b.std() > 0.1


def test_uniform_bias_init_needs_a_trainable_bias():
    with pytest.raises(ValueError):
        Config(hidden_bias="zero", bias_init="uniform").resolved()


def test_run_name_marks_random_bias():
    name = run_name(Config(hidden_bias="trainable", bias_init="uniform", mu=[0, 0]))
    assert "_no-shift_" in name and "_random-bias_" in name
