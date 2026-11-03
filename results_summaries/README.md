# Results summaries — toy models and task streams for loss of plasticity

This folder summarises every experiment of the `toy_models_for_lop` project (code in `plasticity/`, GitHub `gdesideri/toy_models_for_lop`) and the results it produced, as of **2026-11-03**. The project studies **loss of plasticity**: a network trained on a long sequence of tasks gradually becomes unable to learn new ones. It approaches the question from two ends:

1. **Toy models** small enough to track every unit exactly, to find the mechanism (file 01).
2. **Task streams on images** (Shapes3D, MPI3D, CIFAR-10, MNIST) with MLPs, to see when plasticity is actually lost (files 02–07), followed by **acquisition assays** that measure whether a trained network can still learn a new feature compared with a fresh network (files 08–10).

Each experiment is written up in the same six sections: research question, setup, results, references, conclusions, next steps proposed. Next steps are marked *(from the notes / audit / report)* when they come from the project's own documents and *(proposed here)* when they are suggestions added while writing this summary.

The companion project (`~/notebooks`, MNIST task distances, curvature and long MLP streams) is summarised in `~/notebooks/results_summaries/`; how the two connect is in [`cross_project_links.md`](cross_project_links.md).

## Contents

| file | experiments |
|---|---|
| [01_toy_models_and_theory.md](01_toy_models_and_theory.md) | orthogonal-tasks toy (fixed-pool absorption); rotating-target / norm-growth toys; the abandoned gradient-cancellation proof; feature-visualisation notebook |
| [02_shapes3d_streams.md](02_shapes3d_streams.md) | Shapes3D parity feature-switch stream; its width sweep; permuted-pixel stream; single-task probes; representation distance between factors |
| [03_factor_rule_stream.md](03_factor_rule_stream.md) | frozen encoder + persistent head over 800 rules; unfreezing one or all layers |
| [04_class_cluster_pca_streams.md](04_class_cluster_pca_streams.md) | CIFAR-10 class attributes and cluster factors, MNIST cluster factors, Shapes3D cluster control, PCA-cosine benchmark, long-training control (G1) |
| [05_shared_push_validation.md](05_shared_push_validation.md) | input centring / offsets / lr / leaky ReLU / width tests of the "shared push" explanation (E1–E4, M1, W1) |
| [06_memorization_and_representation_settings.md](06_memorization_and_representation_settings.md) | the two self-contained settings (no results on the cluster; known metric issues) |
| [07_mpi3d_streams.md](07_mpi3d_streams.md) | MPI3D-realistic factor streams and width sweep |
| [08_feature_acquisition_result1.md](08_feature_acquisition_result1.md) | Result 1: output remapping vs fixed representation vs encoder adaptation |
| [09_history_acquisition_result2.md](09_history_acquisition_result2.md) | Result 2: which training histories slow later feature acquisition |
| [10_checkpoint_acquisition_pilot.md](10_checkpoint_acquisition_pilot.md) | aged stream checkpoints vs fresh networks on held-out factors |
| [cross_project_links.md](cross_project_links.md) | how these results connect to the `notebooks` project |

Supporting folders:

| folder | what |
|---|---|
| [`figures/`](figures/) | 33 figures: 29 copied from the runs (prefix = the file that cites them) plus 4 made for this summary (`02_width_sweep_collapse`, `04_streams_vs_fresh`, `08_feature_acquisition_gaps`, `09_history_acquisition_deficits`) |
| [`csv/`](csv/) | summary tables: `stream_runs.csv` / `stream_groups.csv` (every task-stream run on the cluster), the four tables behind the new figures, and copies of the acquisition comparisons, contrasts and reports |
| [`checkpoints/`](checkpoints/) | 27 selected checkpoints (109 MB); see [`checkpoints/README.md`](checkpoints/README.md) |
| [`scripts/`](scripts/) | `summarize_streams.py` (builds the stream tables), `make_figures.py` (the 4 new figures and their tables), `collect_artifacts.sh` (copies everything else) |
| [`MANIFEST.tsv`](MANIFEST.tsv) | every copied file: destination, absolute source path, size |

## Headline findings

1. **Streams over fixed images whose labels change can lose trainability completely, through ReLU unit death.** On Shapes3D (parity feature-switch stream) a width-30 MLP is at chance within ~50 tasks with its first layer dead; MPI3D streams do the same; the PCA-cosine MNIST "upper half" stream dies at tasks 45–75 with its second layer dead. → [02](02_shapes3d_streams.md), [07](07_mpi3d_streams.md), [04 E5](04_class_cluster_pca_streams.md)
2. **Width postpones the collapse, it does not prevent it:** on Shapes3D the collapse task grows ~1.06 tasks per hidden unit (46 at width 30 → 235 at width 210), and every width ends at chance with its first layer dead. → [02 E2](02_shapes3d_streams.md)
3. **The same stream constructions on CIFAR-10 lose generalisation, not trainability:** train accuracy 1.000, test 4–6 points below a fresh network, zero dead units; training one task for as long as the whole stream does not reproduce the drop. MNIST cluster streams lose nothing. → [04 E1–E3](04_class_cluster_pca_streams.md)
4. **Centring the inputs prevents the Shapes3D parity-stream collapse**, at least to task 85 where the run stopped (0.999 accuracy, no dead first-layer units, vs chance by tasks 41–55 uncentred) — the strongest support for the "shared push" explanation, with caveats. → [05](05_shared_push_validation.md)
5. **A frozen, well-trained encoder with one persistent head re-learns 800 rule changes with no loss; letting the encoder adapt makes the stream collapse, and adapting more of it makes it collapse faster.** → [03](03_factor_rule_stream.md)
6. **The cleanest toy result is absorption on a finite training pool:** units silent on every training point receive zero gradient and never return, while remaining active on the population (Mac-only data). → [01 E1](01_toy_models_and_theory.md)
7. **The acquisition assay separates three regimes** (output remapping, learning through fixed features, encoder adaptation) and the separation replicates on untouched seeds; on images, encoder adaptation is clearly needed only for narrow encoders. → [08](08_feature_acquisition_result1.md)
8. **Renewed heavy label noise is the history that most slows later feature acquisition (+9 to +11 points); the same noise held fixed is harmless.** Deficits shrink with width and halve by 8,000 updates. → [09](09_history_acquisition_result2.md)
9. **Aged stream checkpoints lose frozen-feature information about held-out factors but full adaptation recovers it completely**, with only a transient slowdown. → [10](10_checkpoint_acquisition_pilot.md)

## Where the project stands (from the 2026-10-28/29 audits and reports)

`reports/Plasticity project publication strategy.md` (with its research notes) and `reports/Plasticity preservation evidence survey.md` assess the project as a strong experimental platform whose image-stream evidence does not yet show a general loss-of-plasticity effect, and whose cleanest positive result (file 01 E1) is a finite-pool toy. Close prior work already covers much of the mechanism (Lyle et al. on pre-activation drift and its causes; an ICML 2026 paper on spectral collapse and fast Gram directions; a May 2026 preprint predicting future plasticity from "optimization readiness"). The report proposes narrowing the project to one prospective question — **can the geometry of future tasks predict when a plasticity intervention will help, fail, or harm useful transfer?** — with a 1–2-week falsification pilot (paired checkpoint branches into historical / fresh / refreshed pools, surgical unit repairs against shams, strong learning-rate controls, a locked held-out confirmation) and a go/no-go threshold before committing 6–10 weeks. Tentative title: "Task Geometry Predicts When Plasticity Repairs Work"; the report states explicitly that none of its proposed results have been obtained. Results 1 and 2 (files 08, 09) and the checkpoint pilot (file 10) are the first steps on that path; the cluster runs in files 02–05 sharpen it, since several of them (collapse on Shapes3D/MPI3D/MNIST upper half, the centring rescue) are stronger than the Mac-only data the audit saw.

## Experiment index and data availability

GPU nodes see only `$HOME` and `/hnvme`; `$WORK` (`/home/atuin/...`) and `$HPCVAULT` (`/home/vault/...`) are for archive and login-node analysis.

| experiment | file | status | data |
|---|---|---|---|
| orthogonal-tasks toy, rotating/norm toys | 01 | complete (toy), missing (rotating/norm) | **Mac only** |
| Shapes3D parity feature stream, permuted-pixel stream, probes, repr distance | 02 | complete | `$HPCVAULT/plasticity_results/shapes3d_*`, `sgd_lr_probe` |
| Shapes3D width sweep | 02 E2 | truncated at tasks 620–641 (all collapsed) | **only on `/hnvme`, expires 2026-12-29** |
| factor-rule stream (6 variants) | 03 | complete | `$HPCVAULT` (base) and `$WORK` (10-epoch unfreeze) |
| CIFAR-10 class / cluster, PCA-cosine streams | 04 | complete | `$WORK/plasticity_results/` (streams), `$HPCVAULT` (single-task baselines, G1) |
| MNIST cluster streams | 04 E3 | **incomplete** (~285/600 tasks) | `$HPCVAULT` |
| Shapes3D cluster stream (control) | 04 E4 | complete | `$WORK`; 100-epoch subset in `plasticity/results/` |
| shared-push validation E1–E4 | 05 | **stopped at 75–111 tasks** of 600/800 | `$HPCVAULT/plasticity_results/e*_*` |
| M1 leaky, W1 width | 05 | complete | `$WORK` |
| memorisation / representation settings | 06 | no runs on the cluster | — |
| MPI3D streams and width sweep | 07 | complete | `$HPCVAULT/*.tar` (one tar per width); single-task baselines unpacked |
| Result 1 feature acquisition | 08 | discovery + synthetic confirmation done; Shapes3D confirmation pending | `$HPCVAULT/plasticity_results/feature_acquisition/` |
| Result 2 history acquisition | 09 | discovery done; confirmations pending | `$HPCVAULT/plasticity_results/history_acquisition/` |
| checkpoint-acquisition pilot | 10 | pilot done (1 history) | `$HPCVAULT/plasticity_results/checkpoint_acquisition/` |

**Moved on 2026-11-03:** the acquisition result folders (feature, history and checkpoint acquisition and their smoke tests, 58 GB) were moved from `plasticity/results/` to `$HPCVAULT/plasticity_results/` to bring `$HOME` back under quota; `plasticity/results/<name>` are now symlinks, so `results_dir` and login-node analysis are unaffected, but **GPU jobs cannot read them** (Result 1/2 confirmations lock settings from the discovery runs).

## Sources and how much to trust them

- **Computed for this summary** from the runs' own files (`args.json`, `task_*/summary.json`, `task_end_metrics.csv`, acquisition `comparisons.csv` / `acquisition_stats.csv`, `alive_units*.json`): every stream number in files 02–07 and the tables behind the four new figures.
- **Written by the analysis code** in each acquisition run (`interpretation.md`, `report.md`).
- **The project's audits and reports** (`reports/`, `research_notes/`), written on the Mac on 2026-10-28/29 — authoritative for the toy models (file 01), whose data is not on the cluster.
- **Prior-session notes** (Claude's memory for this project, `~/.claude/projects/-home-hpc-v170be-v170be10-toy-models-for-lop/memory/`), cited by name, for facts not stored with the runs (job ids, unit-health measurements made post hoc, the user's own interpretations).

## Glossary

- **Stream:** one network trained on a sequence of tasks, carrying its weights (and head) from task to task. **Single-task baseline / fresh network:** a network trained from initialisation on one task for the same budget.
- **Feature-switch / factor stream:** same images every task, label = a function of a different factor each task (output drift). **Permuted-pixel stream:** fixed label rule, fresh pixel permutation per task (input drift).
- **Factors:** Shapes3D's six generative factors (floor hue, wall hue, object hue, size, type/shape, azimuth); MPI3D's seven; **cluster factors** = k-means clusterings in pixel-PCA space (not true generative factors); **PCA-cosine factors** = binned cosine similarity to principal directions.
- **Label rules:** `parity` (factor value mod 2), `upper_half` / `binary_target` (value in the upper half of its range).
- **mlp_LxW:** L hidden ReLU layers of width W. **K (factor-rule stream):** number of encoder layers left trainable.
- **Dead unit:** a ReLU unit whose pre-activation is ≤ 0 on every measured training image (exact for ReLU; for leaky ReLU it is only a sign, not death). **Effective rank:** entropy rank of the last hidden layer's feature matrix.
- **Collapse task:** first task whose 10-task moving average of test accuracy is below 0.75 (used in files 02 and 04).
- **Shared push / push-over-spread:** the norm of the mean input image relative to the per-pixel spread; large on Shapes3D (~1,846) and MPI3D, small on CIFAR-10 (31.8).
- **Acquisition assay (files 08–10):** from a checkpoint, learn a destination task by full adaptation, by frozen decoders on the features, or from a fresh network; **gap** = full minus best frozen decoder; **deficit** = fresh minus checkpoint test accuracy over a fixed update budget (positive = worse than fresh).
- **Discovery / confirmation:** discovery runs select learning rates and settings on seeds 0–5; confirmation reruns on untouched seeds (100–111) with those choices locked.
