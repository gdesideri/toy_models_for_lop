#!/bin/bash -l
#SBATCH --job-name=unit-health
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Measures the hidden units of runs that did not log them (every run before
# run_tasks_stream --health_every): visualization_scripts/unit_health.py --compute, from the
# end-of-task checkpoints, on the same data and probe as the logging. One array index per
# (stream directory, seed): index I -> STREAMS[I / #seeds], SEEDS[I % #seeds].
#
#   STREAMS="results/a results/b"   stream results directories (default: the launch_all.sh streams;
#                                   add the older Shapes3D streams and the MNIST streams by path,
#                                   e.g. "$HPCVAULT/plasticity_results/shapes3d_stream_features_sgd")
#   SEEDS="0 1 2"   EVERY=5   (measure every 5th task and the last)
#
# Usage (from plasticity/): array size = #streams x #seeds
#   sbatch --array=0-8 experiments/launch_scripts/submit_unit_health.sh
#   STREAMS="results/mnist_cluster_stream_mlp_2x30" sbatch --array=0-2 experiments/launch_scripts/submit_unit_health.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f experiments/run_tasks_stream.py ]] || { echo "ERROR: submit from the plasticity/ directory" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

read -r -a STREAMS_LIST <<< "${STREAMS:-results/shapes3d_cluster_stream_mlp_2x30 results/cifar10_class_stream_mlp_2x120 results/cifar10_cluster_stream_mlp_2x120}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
I="${SLURM_ARRAY_TASK_ID:-0}"
N=$(( ${#STREAMS_LIST[@]} * ${#SEEDS[@]} ))
(( I < N )) || { echo "ERROR: array index $I outside 0-$(( N - 1 ))" >&2; exit 1; }
STREAM="${STREAMS_LIST[$(( I / ${#SEEDS[@]} ))]}"
SEED="${SEEDS[$(( I % ${#SEEDS[@]} ))]}"
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))}"

echo "unit health: $STREAM seed $SEED, every ${EVERY:-5} tasks"
CMD=("$UV" run --no-sync python -u -m visualization_scripts.unit_health --compute "$STREAM" --seed "$SEED"
     --every "${EVERY:-5}" --workers "$WORKERS" --device cuda)
if [[ -n "${DRY_RUN:-}" ]]; then echo "${CMD[*]}"; exit 0; fi
srun "${CMD[@]}"
