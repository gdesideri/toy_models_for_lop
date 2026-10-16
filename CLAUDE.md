# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code on loss of plasticity (LoP) in continual learning: toy regression models (NTK, rotating targets, norm growth) and larger task-stream experiments on MNIST / CIFAR-10 / Shapes3D. All code lives in `plasticity/`, a `uv` project (Python 3.14 pinned in `.python-version`, `>=3.11` required). Tests: `uv run pytest` from `plasticity/` (pytest is a uv dev dependency; there is no linter or build step).

## Commands

Run everything from `plasticity/`. Modules import each other as top-level packages (`from models.factory import ...`), so run scripts as modules from that directory:

```bash
cd plasticity
uv sync                                                        # set up .venv
uv run python -m experiments.run_tasks_stream --dataset_kind shapes3d --target_factor 5 \
    --label_rule parity --model_kind mlp_2x30 --n_tasks 10 --num_epochs 10 --batch_size 256
uv run python -m experiments.run_factor_rule_stream --n 256 --seed 0 ...
uv run python -m memorization_plasticity.run --dataset cifar10 --arch mlp_2x512 --p_change 0.1 --L 500
uv run python -m representation_plasticity.run --stream ladder --arch mlp_2x512 --switch_every 2000
uv run python -m visualization_scripts.<script>                # most plots; some take --compute
uv run python visualization_scripts/toy_models/<script>.py     # toy-model scripts: config is set at the top of the file
```

Each script's module docstring documents its exact usage, outputs and on-disk layout — read it before running or changing a script.

Unit health (dead / rarely active units, effective rank) is logged per task by `run_tasks_stream` (`--health_every`, default 1) and summarised with `uv run python -m visualization_scripts.unit_health --runs name=<stream dir> ...`. `uv run python -m experiments.quick_unit_health --dataset shapes3d --conditions base center=1` is a fast in-memory pilot (minutes, local), not for final numbers.

Shapes3D must be downloaded once (GPU nodes cannot download):
`uv run python -c "from data.download import download_shapes3d; download_shapes3d()"`. Data is cached under `data/_cache/` or `$PLASTICITY_DATA_DIR`.

### Cluster (Slurm)

`launch_all.sh` (class-attribute / cluster-factor streams) and `launch_validation.sh` (the shared-push experiments: centred inputs, offsets, lr, leaky ReLU, width) submit whole experiment sets through `submit_stream_array.sh`; run them with `bash`, `DRY_RUN=1` first. `submit_unit_health.sh` measures units of older runs post hoc.

`experiments/launch_scripts/submit_*.sh` are Slurm array jobs (H100 partition, account `v170be`). Submit from `plasticity/`: `sbatch experiments/launch_scripts/<script>.sh`. They are configured with environment variables (e.g. `SEEDS`, `MODEL_KIND`, `N_TASKS`, `RUN_TAG`, `PERM`, `TASK_FACTORS`), one array index per seed/config, run `uv run --no-sync`, and expect `.venv` to exist already. Long streams use `--deadline_h` + `--resume` and the script resubmits itself only if the job made progress. (Some usage comments still say `experiments/submit_*.sh`; the scripts are in `launch_scripts/`.)

## Architecture

- **`models/factory.py`** — the single entry point for networks: `build_model(arch, ...)` with `mlp_<layers>x<width>` (any size), `cnn`, and scalar-output toy archs (`minimal*`, `shallow_<h>`, `deep_<h>`, `linear`); `build_optimizer` (`sgd` = momentum 0 by default, `adam`). EWC is a loss penalty (`models/ewc.py`), not an optimizer. Classifiers expose `model.head` (`reset_head`, `freeze_body`).
- **`data/`** — `download.py` lazily fetches datasets (nothing on import); `factory.py` builds train/test splits (`build_dataset`, `dataset_spec`, `build_loaders`) and Shapes3D labelling (`target_factor`, `LABEL_RULES` such as `parity`, `binary_target`).
- **`tasks/`** — task definitions: `permuted_pixels` (input drift), `feature_streams` / `factor_rules` (Shapes3D label/feature switches = output drift), `randomized_labels` (memorization stream), `toy_regression` (toy-model tasks).
- **`experiments/`** — runners. `run_basic_task_training.train` is the shared training loop that `run_tasks_stream` reuses across tasks. Arguments for these runners live in `utils/args.py`.
- **`memorization_plasticity/`, `representation_plasticity/`** — self-contained settings, each with its own `run.py` and `metrics.py`.
- **`src/`** — toy-model utilities: empirical NTK/Jacobians (`ntk.py`), MSE training (`train.py`), probe bank of fresh-model comparisons (`probe.py`).
- **`utils/`** — `results.run_dir_for` derives the run directory name from params (dataset, model, perm %, Shapes3D label/factors, seed) — keep labels in the name so sweeps don't overwrite each other; `checkpoints` stores `model_kwargs` so `load_model_from_checkpoint` rebuilds a net via the factory; `stream_probe` reloads task streams (per-task checkpoints and pixel permutations) for post-hoc analysis; `evaluation` has alive/dead ReLU unit measurements.
- **`visualization_scripts/`** — analysis and plots over saved runs. `common.py` holds the shared style (`plotstyle` Nature preset, Okabe-Ito palette) and results lookup: `results_dir` searches `plasticity/results`, then `$HPCVAULT/plasticity_results`, then `$WORK/plasticity_results`; stream result folder names (e.g. `shapes3d_stream_perm100_sgd`) are constants there. Expensive measurements are split into a GPU `--compute` pass that writes a per-seed JSON cache into the run directory and a CPU plotting pass that reads it.

## Conventions

- Outputs go to `plasticity/results/` (gitignored), Slurm logs to `plasticity/logs/`.
- Task streams carry weights from task to task; each task dir holds `history.json`, `metrics.csv`, `summary.json`, its pixel permutation, and `checkpoints/epoch_<e>.pt`. Snapshots store RNG state so `--resume` is exact.
- Default experimental budget: plain SGD (momentum 0), 10 epochs per task, batch 256, lr 0.001, seeds 0–2.
- Design specs and implementation plans go in `docs/superpowers/{specs,plans}/` (`docs` is in `.gitignore`, though the existing files are tracked).
