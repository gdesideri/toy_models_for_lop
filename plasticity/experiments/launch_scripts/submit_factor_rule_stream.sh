#!/bin/bash -l
#SBATCH --job-name=s3d-factor-rule
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# run_factor_rule_stream.py on Shapes3D: pretrain an mlp_3x<N> on all six factors, freeze
# it, then train ONE persistent Linear(N, 2) head through 800 binary factor-rule tasks
# (task t labelled by the upper half of factor t % 6). Plain SGD, momentum 0, everywhere.
#
# One array task = one seed (split, init, batches).
#
# Usage:
#   cd plasticity && sbatch experiments/launch_scripts/submit_factor_rule_stream.sh
#   N=512 STEPS=1000 sbatch experiments/launch_scripts/submit_factor_rule_stream.sh
#   N_SAMPLES=60000 EPOCHS=5 sbatch --array=0 --time=01:00:00 experiments/launch_scripts/submit_factor_rule_stream.sh  # short probe
#
# Reuse an earlier run's pretrained encoder (seed k loads ENCODER_DIR/factorrule_..._seed<k>/checkpoints/encoder.pt,
# copied, never modified; must be visible from the GPU nodes, so not /home/vault) and train the
# top UNFREEZE_LAYERS hidden layers with the head (1: the layer feeding the head, 3: the full network):
#   ENCODER_DIR=results/shapes3d_factor_rule_stream UNFREEZE_LAYERS=1 N=30 EPOCHS=10 LR_PRETRAIN=0.001 \
#       OUTDIR=results/shapes3d_factor_rule_stream_unfreeze_last sbatch experiments/launch_scripts/submit_factor_rule_stream.sh

set -euo pipefail

_find_root() {
    local d="$1"
    while [[ "$d" != "/" ]]; do
        [[ -f "$d/experiments/run_factor_rule_stream.py" ]] && { echo "$d"; return 0; }
        d="$(dirname "$d")"
    done
    return 1
}
PROJECT_ROOT="${PROJECT_ROOT:-$(_find_root "${SLURM_SUBMIT_DIR:-$PWD}" \
    || _find_root "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")}" || {
    echo "ERROR: cannot find the project; submit from the plasticity/ directory." >&2; exit 1; }
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
H5="${PLASTICITY_DATA_DIR:-$PROJECT_ROOT/data/_cache}/shapes3d/3dshapes.h5"
[[ -f "$H5" ]] || { echo "ERROR: $H5 is missing and GPU nodes cannot download it." >&2; exit 1; }

read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"

N="${N:-256}"
EPOCHS="${EPOCHS:-30}"
LR_PRETRAIN="${LR_PRETRAIN:-0.003}"
LR_HEAD="${LR_HEAD:-0.01}"
PROBE_EPOCHS="${PROBE_EPOCHS:-10}"
STEPS="${STEPS:-500}"
BATCH="${BATCH:-256}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
WORKERS=$(( WORKERS > 1 ? WORKERS - 1 : 0 ))
OUTDIR="${OUTDIR:-results/shapes3d_factor_rule_stream}"
ENCODER_FROM=""
if [[ -n "${ENCODER_DIR:-}" ]]; then
    ENCODER_FROM="$ENCODER_DIR/factorrule_shapes3d_mlp_3x${N}_perm000_seed${SEED}/checkpoints/encoder.pt"
    [[ -f "$ENCODER_FROM" ]] || { echo "ERROR: $ENCODER_FROM is missing." >&2; exit 1; }
fi

echo "seed=$SEED n=$N epochs=$EPOCHS lr_pretrain=$LR_PRETRAIN lr_head=$LR_HEAD steps/task=$STEPS batch=$BATCH -> $OUTDIR"
echo "encoder_from=${ENCODER_FROM:-<pretrain>} unfreeze_layers=${UNFREEZE_LAYERS:-0}"

srun "$UV" run --no-sync python -u -m experiments.run_factor_rule_stream \
    ${N_SAMPLES:+--n_samples "$N_SAMPLES"} \
    ${ENCODER_FROM:+--encoder-from "$ENCODER_FROM"} \
    --unfreeze-layers "${UNFREEZE_LAYERS:-0}" \
    --n "$N" \
    --pretrain-epochs "$EPOCHS" \
    --lr-pretrain "$LR_PRETRAIN" \
    --lr-head "$LR_HEAD" \
    --probe-epochs "$PROBE_EPOCHS" \
    --steps-per-task "$STEPS" \
    --batch_size "$BATCH" \
    --num_workers "$WORKERS" \
    --seed "$SEED" \
    --output-dir "$OUTDIR" \
    "$@"
