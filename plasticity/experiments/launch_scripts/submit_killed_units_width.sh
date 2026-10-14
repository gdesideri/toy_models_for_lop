#!/bin/bash -l
#SBATCH --job-name=killed-units-width
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Units killed by the optimizer during every task of the feature-switch width sweep
# (see the script). Array index I: the I-th run of the stream dir, sorted by name.
# Then, from a login node:  uv run python -m visualization_scripts.stream_killed_units_width
#
# Usage:  cd plasticity && sbatch experiments/launch_scripts/submit_killed_units_width.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT="${SCRIPT:-visualization_scripts/stream_killed_units_width.py}"
[[ -f "$SCRIPT" ]] || { echo "submit from plasticity/" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
RUNS=($(ls -d results/shapes3d_stream_features_width_sgd/stream_*_seed* | sort))
I="${SLURM_ARRAY_TASK_ID:-0}"
RUN="${RUNS[$I]}"
echo "array task $I -> $RUN"
srun "${UV:-$HOME/.local/bin/uv}" run --no-sync python -u "$SCRIPT" \
    --compute "$RUN" --n_probe "${N_PROBE:-20000}" --device cuda
