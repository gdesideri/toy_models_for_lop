# 01 — Toy models and theory

Small, fully inspectable networks (a handful of ReLU units, 2–3-dimensional inputs) built to isolate mechanisms of loss of plasticity, plus the mathematical claims made about them. Terms are defined in the [README glossary](README.md#glossary).

**Data location.** None of the toy-model outputs (`plasticity/results/orthogonal_tasks/`, the rotating-target and norm-growth runs, `toy_audit_20261028/`) are on the cluster: they live in the copy of the repository on the Mac (`/Users/ariv18uq/Desktop/toy_models_for_lop`). Everything below is taken from the audit written on that machine on 2026-10-28/29 (`research_notes/Plasticity project publication strategy/mechanism_audit.md`, in git), which reconstructed the key numbers from the raw arrays. No toy result here was re-checked on the cluster.

Experiments in this file:

- [E1 — Orthogonal-tasks toy: fixed-pool absorption](#e1--orthogonal-tasks-toy-fixed-pool-absorption)
- [E2 — Rotating-target and norm-growth toys](#e2--rotating-target-and-norm-growth-toys)
- [E3 — The gradient-cancellation ("antipodal") proof](#e3--the-gradient-cancellation-antipodal-proof)
- [E4 — Feature-visualisation notebook](#e4--feature-visualisation-notebook)

---

## E1 — Orthogonal-tasks toy: fixed-pool absorption

**Status:** runs complete on the Mac (31 configurations, 94 seed summaries at the 2026-10-28 audit). **Data:** Mac only. Code on the cluster: `plasticity/orthogonal_tasks/` (package), `experiments/run_orthogonal_tasks.py`, `src/unit_geometry.py`, `tasks/orthogonal_halfspaces.py`; GIF and summary plots added 2026-10-30.

### 1. Research question
In a network small enough to track every unit exactly, how does a ReLU unit stop learning when two half-space tasks alternate, and is the dead unit dead on the whole input population or only on the training sample?

### 2. Setup
- Inputs from a support S = {μ + A·u : ‖u‖ ≤ 1} (here centred, μ = 0); two tasks labelled by half-spaces with normals (e0 ± e1)/√2; tasks alternate in phases.
- Four ReLU units with trainable biases, plain SGD (no momentum), a fixed pool of 2,048 training and 2,048 test points, 10k updates per phase.
- Instrumentation: exact population-support maxima of each unit's pre-activation, train/test activity, analytic full-batch gradients, and a fresh network trained at every phase.

### 3. Results
- **The cleanest positive result of the project.** Phase-end live-unit counts fall 4 → 2 → 1 → 0. At phase 3 continued test accuracy is **49.71%** versus **99.80%** for a fresh network, and it stays at chance.
- Every hidden feature is **exactly zero on every training example** and all empirical gradients vanish. Yet all four units keep a strictly positive maximum pre-activation on the population support ([0.109, 0.022, 0.044, 0.138]); a few held-out examples still activate them (1/2048, 0, 0, 2/2048). Hidden-weight norms are 1.5–6.5, so this is not norm collapse.
- Across the 94 seed summaries: 1,866 phase endpoints, 315 first-death events, **5** deaths certified on the population, 0 norm collapses, **0 revivals**. The runner's own labels call 293 deaths "bias-driven" and 22 "rotation into the inactive cone", but those labels are event classifications, not a causal decomposition.
- **Confound flagged by the audit:** the "thin-x1" (fails) vs "thin-x3" (succeeds) contrast changes the labels, conditioning and initial function as well as the support geometry, so it is not a clean coverage intervention.

### 4. References
- Audit: [`research_notes/.../mechanism_audit.md`](../research_notes/Plasticity%20project%20publication%20strategy/mechanism_audit.md); summary in [`reports/Plasticity project publication strategy.md`](../reports/Plasticity%20project%20publication%20strategy.md)
- Run `3d_AB-diagonal_no-shift_thin-x1_random-bias_4-units_10k-steps_10-switches_live-init/seed0` (Mac)

### 5. Conclusions
- What is shown is **absorption on the finite training pool**: a unit silent on every training point gets zero gradient under plain SGD and stays silent for any later labels. A correct lemma follows: for first-layer ReLUs with fixed training inputs, if w_i·x_n + b_i ≤ 0 for every training x_n, the gradients of w_i, b_i and the readout vanish for any subsequent labels.
- It is not loss of all population-accessible capacity: the units are still alive on the population. Whether fresh samples would rescue the model within a useful budget is untested.
- Input mean alone is not the cause: the strongest collapse has mean-zero inputs.

### 6. Next steps proposed
*(From the audit and the publication-strategy report.)*
1. Paired checkpoint experiment: branch identical checkpoints into the historical pool, an independent pool from the same population, continuously refreshed samples, and a duplicate-of-old-pool control, at pool sizes 512 / 2,048 / 8,192.
2. Measure each unit's activation probability p_i on independent future inputs, and test whether "silent on history, active on future support" predicts which repair helps.
3. A minimal causal repair: zero a silent unit's readout and move only its bias to a chosen activation quantile (old predictions unchanged), against sham and random-unit controls.
4. Reconcile the two inconsistent toy inventories (28 configs / 85 seeds in the cached audit vs 31 / 94 on disk) before publishing any aggregate.

---

## E2 — Rotating-target and norm-growth toys

**Status:** code exists; the expected outputs were **absent** even on the Mac at the audit. **Data:** none found.

### 1. Research question
Does a growing weight norm, or a target that rotates over time, slow learning through the NTK spectrum?

### 2. Setup
`experiments/run_rotating_target.py`, `run_norm_growth_270.py`: four-unit bias-free ReLU regressors, Gaussian inputs, linear targets, Adam.

### 3. Results
No saved local JSON/NTK artefacts were found. The audit lists design problems: scaling the target by 20 changes amplitude, readout, residual size and Adam history together (not only the hidden norm); rotating snapshots omit the readout, and the NTK reconstruction then falls back to a random readout; a width-4 network has at most eight independent Jacobian directions, so many zero NTK eigenvalues are structural.

### 4. References
[`mechanism_audit.md`](../research_notes/Plasticity%20project%20publication%20strategy/mechanism_audit.md), section "Old rotating/norm toy status".

### 5. Conclusions
Not evidence for a norm or NTK obstruction. ReLU rescaling (w → c·w, a → a/c) changes norms and the NTK without changing the function, so norm and NTK statistics cannot be the main claim.

### 6. Next steps proposed
*(From the audit)* Drop as a headline; keep only if linked to an effect validated elsewhere.

---

## E3 — The gradient-cancellation ("antipodal") proof

**Status:** design and plan written 2026-04-22; **the central theorem is false**. **Data:** n/a.

### 1. Research question
Can a symmetric ("antipodal") pair of ReLU units become frozen in function space because their parameter gradients cancel, removing two units of capacity?

### 2. Setup
Notebook design `docs/superpowers/specs/2026-04-22-gradient-flat-proof-design.md` and plan `docs/superpowers/plans/2026-04-22-gradient-flat-proof.md` (Parts A–C: derivation, symmetric-pair network, loading phases).

### 3. Results
Counterexample (audit): x ∈ {−1, +1}, y = |x|, w₊ = 0.5, w₋ = −0.5, readouts 1, biases 0. The two hidden-weight gradients (−0.5, +0.5) sum to zero, yet one gradient step changes the function to (0.5 + 0.5η)|x|. Under full-batch gradient flow the function changes as df/dt = −J Jᵀ ∇_f L; whether two parameter-block gradients sum to zero is irrelevant.

### 4. References
[`mechanism_audit.md`](../research_notes/Plasticity%20project%20publication%20strategy/mechanism_audit.md), section "Which mathematical claims survive"; the spec and plan above.

### 5. Conclusions
Abandon the theorem. The two correct statements that survive (fixed-pool absorption, E1; the exact population-support condition w·μ + ‖Aᵀw‖ + b ≤ 0) are interpretive tools, not contributions.

### 6. Next steps proposed
*(From the audit)* Use the NTK/Jacobian already implemented in `src/ntk.py` for any statement about functional stasis.

---

## E4 — Feature-visualisation notebook

**Status:** in progress (prior-session note). **Data:** reads `results/representation/repr_ladder_mlp_2x512_relu_ln0_optadam_sw25000_s0`, which is not on the cluster (the vault's `representation/` folder is empty).

### 1. Research question
Do individual first-layer MLP units specialise, drift or die over training, as seen by gradient-ascent feature visualisation?

### 2. Setup
`plasticity/notebooks/feature_understanding.ipynb`: Lucent `render_vis` on 8 of the 512 first-layer units, at 6 evenly spaced snapshots, pixel parametrisation at 64×64.

### 3. Results
No saved output on the cluster.

### 4. References
Prior-session note "project_feature_understanding_notebook" (Claude memory, toy_models project).

### 5. Conclusions
None yet.

### 6. Next steps proposed
*(Proposed here)* Point it at a run that exists on the cluster with per-task checkpoints (e.g. the Shapes3D width sweep before it expires, file 02).
