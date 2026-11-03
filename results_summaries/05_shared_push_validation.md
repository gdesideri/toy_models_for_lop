# 05 — Validation of the "shared push" explanation (E1–E4, M1, W1, G1)

**Status:** launched 2026-10-16 (`launch_validation.sh`); **most streams stopped early** — the E1–E4 streams reached 75–111 of their 600 (or 800) tasks and were not resumed; M1 and W1 completed 600 tasks; G1 is reported in file 04. **Data:** `$HPCVAULT/plasticity_results/e{1,2,3,4}_*`, `g1_*`; `$WORK/plasticity_results/{m1,w1}_*`. There is no written analysis of this sweep in the repository; everything below is computed for this summary from the runs' task summaries ([`csv/stream_runs.csv`](csv/stream_runs.csv), [`csv/stream_groups.csv`](csv/stream_groups.csv)).

## 1. Research question
The streams that lose trainability with unit death are on Shapes3D; the same constructions on CIFAR-10 and MNIST do not (file 04). Shapes3D images share a very large common component: their mean image dominates their spread ("push/spread" ≈ 1,846, vs 31.8 for CIFAR-10). The hypothesis ("shared push"): a large shared input component pushes every first-layer pre-activation the same way at every gradient step, so units are driven to silence together. Does removing the push prevent the collapse, and does adding it to CIFAR-10 create one?

## 2. Setup
Plain SGD, lr 0.001, batch 256, seeds 0–2, 100 epochs per task, mlp_2x30 unless stated, per-task unit health logged.

| arm | change | data / stream |
|---|---|---|
| E1 | centre the inputs (subtract the training mean image): push → 0 | Shapes3D cluster; the original Shapes3D parity feature stream (800 tasks × 10 epochs); MNIST cluster at 2x30 and 2x120 |
| E2 | give CIFAR-10 a Shapes3D-sized push (input offset 0.9113: push/spread ≈ 1,849) | CIFAR-10 cluster, plus its 2x30 baseline |
| E3 | doses: Shapes3D centred by 0.5 / 0.9 (push/spread ≈ 461 / 18); CIFAR-10 offsets for 200 / 600 | |
| E4 | push × lr matched to MNIST: Shapes3D at lr 1e-4; CIFAR-10 at lr 1e-2 | |
| M1 | leaky ReLU (silent units keep a gradient) | Shapes3D cluster |
| W1 | width 120 | Shapes3D cluster |

Most single-task baselines (`*_single`) ran alongside.

## 3. Results
Mean test accuracy over fixed task windows (3 seeds; "–" = the stream stopped before that window):

| stream | push/spread | tasks 0–9 | tasks 65–74 | tasks 90–99 | dead L1 / L2 at last task |
|---|---|---|---|---|---|
| **Shapes3D parity features, 2x30, baseline** (file 02 E1) | ~1,846 | 0.983 | **0.528** | 0.511 | 28–30 / 15–30 dead of 30 by task 100 |
| **E1: same, centred** | 0 | 0.938 | **0.999** | – (stopped at 85) | **0 / 2.3** |
| Shapes3D cluster, 2x30, baseline (file 04 E4) | ~1,846 | 0.950 | 0.927 | 0.923 | (~28 / ~24 by task 500) |
| E1: centred | 0 | 0.951 | 0.945 | – (86) | 0 / 1.3 |
| E3: centred × 0.5 | 461 | 0.953 | 0.946 | – (77) | 3.7 / 2.3 |
| E3: centred × 0.9 | 18 | 0.952 | 0.947 | – (77) | 0 / 0 |
| E4: lr 1e-4 | 1,846 | 0.915 | 0.915 | – (82) | 1.7 / 8.0 |
| M1: leaky ReLU | 1,846 | 0.948 | 0.934 | 0.929 | (600 tasks: 0.907 last visits; sign-negative 10.3 / 29.7) |
| W1: width 120 | 1,846 | 0.955 | 0.956 | 0.956 | (600 tasks: 0.881 last visits; **105.7 / 80.3 of 120**) |
| CIFAR-10 cluster, 2x30, baseline (E2) | 31.8 | 0.842 | 0.860 | 0.860 | 0 / 0 |
| E2: offset, push/spread 1,849 | 1,849 | 0.814 | 0.813 | 0.831 | 0 / 0 |
| E3: offset 200 / 600 | 200 / 600 | 0.838 / 0.830 | 0.847 / 0.845 | – | 0 / 0 |
| E4: lr 1e-2 | 31.8 | 0.887 | 0.875 | – (84) | 0 / **10.3**, effective rank 13.5 → 1.8 |
| MNIST cluster, 2x30 (file 04 E3) | – | 0.875 | 0.873 | 0.880 | (not logged) |
| E1: MNIST centred, 2x30 | 0 | 0.882 | 0.898 | 0.902 | 0 / 0 |

- **Centring prevents the parity-feature collapse.** With identical settings otherwise (mlp_2x30, lr 0.001, 10 epochs, 800-task parity stream), the uncentred stream is at chance by tasks 41–55; the centred one is at 0.999 with no dead first-layer units at task 85, where it stopped.
- **On the cluster stream the window is too short:** the uncentred baseline only loses ~3 points by task 99, and the centred, dosed and low-lr arms stopped before that.
- **Adding a Shapes3D-sized push to CIFAR-10 does not kill units within 100 tasks;** it lowers accuracy by ~3 points from the first task and shrinks effective rank faster (10.6 → 6.4 vs 12.8 → 10.8).
- **A 10× larger lr on CIFAR-10 kills a third of layer 2 and collapses its rank (→ 1.8) within 84 tasks.**
- **Width 120 does not prevent unit death** on the Shapes3D cluster stream: 106 of 120 first-layer units are dead at task 599, yet accuracy is still 0.881 (vs 0.859 at width 30).
- **Leaky ReLU** ends higher than the ReLU baseline (0.907 vs 0.859). Its "dead" counts use the sign of the pre-activation, which is not functional death for a leaky unit.

## 4. References
- Launcher with the arm definitions: `experiments/launch_scripts/launch_validation.sh`; offsets from `python -m data.input_shift --dataset cifar10 --target_ratio 200 600 1849`
- Input statistics (`shared_push`, `push_over_spread`) are logged in every task summary's `input_stats`
- Data: [`csv/stream_runs.csv`](csv/stream_runs.csv) (groups `e1_*` … `w1_*`)
- Audit caveats: [`research_notes/.../evidence_audit.md`](../research_notes/Plasticity%20project%20publication%20strategy/evidence_audit.md) §2

## 5. Conclusions
- The strongest support for the shared-push explanation is E1 on the parity feature stream: removing the shared component turns a collapse at ~50 tasks into near-perfect accuracy at 85. But the evidence audit's caveat applies: centring changes both the initial pre-activations and the optimisation coordinates, so one intervention cannot identify the shared push as the unique cause; and on CIFAR-10 a large push alone did not reproduce the collapse within 100 tasks.
- Learning rate is at least as strong a lever as the push (E4 CIFAR-10), and width is not a cure for unit death (W1).

## 6. Next steps proposed
- *(Proposed here)* Resume the E1–E4 streams to 600/800 tasks (`--resume`; all write per-task health). The single most informative one is E1 on the parity feature stream: does centring prevent the collapse, or only delay it like width does (file 02 E2)?
- *(From the audit)* Add a function-matched control: transform the uncentred network's first-layer bias (b' = b − Wμ) so that centred and uncentred runs start from the same function, separating input geometry from initialisation.
