#!/bin/bash -l
#SBATCH --job-name=unit-activity
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Replays tasks t and t+1 of a task stream from its checkpoints and records every
# hidden unit's activity after each epoch (experiments/replay_stream_unit_activity.py).
# Then plot from a login node: uv run python -m visualization_scripts.stream_unit_activity
#
# Usage:  cd plasticity && sbatch experiments/submit_replay_unit_activity.sh
#         RUN_DIR=results/.../stream_..._seed1 TASKS="20 70" sbatch experiments/submit_replay_unit_activity.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f experiments/replay_stream_unit_activity.py ]] || { echo "submit from plasticity/" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
RUN_DIR="${RUN_DIR:-$(ls -d results/shapes3d_stream_perm100_sgd/stream_*_seed0)}"
TASKS="${TASKS:-20 70 300 400}"
WINDOW="${WINDOW:-2}"                    # consecutive tasks replayed from each start task
echo "replaying $WINDOW tasks from each of $TASKS of $RUN_DIR"
srun "${UV:-$HOME/.local/bin/uv}" run --no-sync python -u -m experiments.replay_stream_unit_activity \
    --run_dir "$RUN_DIR" --tasks $TASKS --window "$WINDOW" --device cuda
