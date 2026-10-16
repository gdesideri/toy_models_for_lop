#!/bin/bash
# Solvability check for the robot-arm factor benchmark: before any stream, every task of it
# alone on a fresh network. Is each factor's binary label learnable by an MLP from pixels?
# Through submit_stream_array.sh (MODE=factors SINGLE=1): one array task per (factor, seed),
# plain SGD (lr 0.001, batch 256), 100 epochs (the stream gives a task 10: history.json has
# the test accuracy after every epoch, so both budgets are read from the same run).
#
#   <dataset>_factor_single_<model>   dataset in DATASETS, model in MODELS
#       mpi3d_realistic  7 factors x 3 seeds = array 0-20
#       isaac3d          9 factors x 3 seeds = array 0-26
#   mlp_2x30 is the stream's network; mlp_2x120 tells a capacity limit from an unlearnable label.
#
# Labels (data/robot3d.py): parity of the factor index (even -> 1), except the robot arm's
# position -- MPI3D horizontal_axis / vertical_axis, Isaac3D robot_x / robot_y -- labelled by a
# threshold in the middle (index >= K/2). Factors with an odd number of values are unbalanced
# (3 values: 2/3 vs 1/3; robot_y: 60/40): read accuracies against the majority class (the report does).
#
# Before the first launch, on a node that can download (cd plasticity; the GPU nodes cannot):
#   uv sync
#   uv run python -m data.robot3d prepare mpi3d_realistic                  # downloads ~12 GB, keeps 480k
#   uv run python -m data.robot3d prepare isaac3d --source /path/to/Isaac3D_down128
#       (the 128x128 release, downloaded by hand from github.com/NVlabs/High-res-disentanglement-datasets)
#
# Usage (from plasticity/):
#   DRY_RUN=1 bash experiments/launch_scripts/launch_robot3d_single.sh    # print the sbatch commands only
#   bash experiments/launch_scripts/launch_robot3d_single.sh
#   DATASETS=isaac3d MODELS=mlp_2x30 bash experiments/launch_scripts/launch_robot3d_single.sh
# Then:
#   uv run python -m visualization_scripts.single_task_solvability \
#       --runs results/mpi3d_realistic_factor_single_mlp_2x30 results/isaac3d_factor_single_mlp_2x30 ...

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."           # plasticity/
mkdir -p logs
SCRIPT=experiments/launch_scripts/submit_stream_array.sh
DATASETS="${DATASETS:-mpi3d_realistic isaac3d}"
MODELS="${MODELS:-mlp_2x30 mlp_2x120}"
EPOCHS="${EPOCHS:-100}"
COMMON=(OPTIMIZER=sgd LR=0.001 BATCH=256 SEEDS="0 1 2" HEALTH_EVERY=1 MODE=factors SINGLE=1 EPOCHS="$EPOCHS")

for dataset in $DATASETS; do
    case "$dataset" in
        mpi3d_realistic) n_factors=7 ;;
        isaac3d)         n_factors=9 ;;
        *) echo "ERROR: DATASETS: use mpi3d_realistic and/or isaac3d, got $dataset" >&2; exit 1 ;;
    esac
    for model in $MODELS; do
        name="${dataset}_factor_single"
        cmd=(env "${COMMON[@]}" DATASET="$dataset" MODEL_KIND="$model" RUN_TAG="${name}_${model}"
             sbatch --job-name="$name" --array="0-$(( 3 * n_factors - 1 ))" --time=04:00:00 "$SCRIPT")
        if [[ -n "${DRY_RUN:-}" ]]; then
            printf '%q ' "${cmd[@]}"; echo
        else
            echo "${name}_${model}: $("${cmd[@]}")"
        fi
    done
done
