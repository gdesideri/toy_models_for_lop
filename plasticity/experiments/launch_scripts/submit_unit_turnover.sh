#!/bin/bash -l
#SBATCH --job-name=unit-turnover
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# R / K / O unit turnover of every task of the task streams (see the script), measured on
# a probe of each task.s own training data.
# Array index I: stream I / 3 (0 = permuted pixels, 1 = feature switches), seed I % 3.
# Then, from a login node:  uv run python -m visualization_scripts.stream_unit_turnover

#
# Usage:  cd plasticity && sbatch visualization_scripts/submit_alive_units.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f visualization_scripts/stream_unit_turnover.py ]] || { echo "submit from plasticity/" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
STREAMS=(results/shapes3d_stream_perm100_sgd results/shapes3d_stream_features_sgd)
I="${SLURM_ARRAY_TASK_ID:-0}"
STREAM="${STREAMS[$(( I / 3 ))]}"
SEED=$(( I % 3 ))
echo "array task $I -> $STREAM seed $SEED"
srun "${UV:-$HOME/.local/bin/uv}" run --no-sync python -u -m visualization_scripts.stream_unit_turnover \
    --compute "$STREAM" --seed "$SEED" --n_probe "${N_PROBE:-20000}" --device cuda
