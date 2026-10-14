#!/bin/bash -l
#SBATCH --job-name=s3d-reprdist
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append

# calculate_repr_distance.py on Shapes3D, 100% permuted pixels, 10 pairs of tasks.
#
# One array task = one seed = one pair of tasks. The seed draws the train/test
# split and then two independent full pixel permutations (task 1, task 2), so
# the 10 runs are 10 different pairs. Each run:
#   1. trains a model on task 1 and a model on task 2, from scratch
#   2. zero-shot: loss/acc of the task-1 model on task 2's training set
#   3. resets the task-1 model's head, freezes its body, retrains the head on task 2
#   4. repr distance = test acc(task-2 model) - test acc(task-1 body + new head)
#
# Model: mlp_2x30. Label: object_azimuth, either as 15 classes (default) or,
# with LABEL_RULE=parity, as a binary task -- even azimuth index -> class 1,
# odd -> class 0 (8 vs 7 of the 15 bins).
#
# Caveat: the distance is only clean when both from-scratch models learn their
# task. In the width sweep mlp_2x30 solved 15-class azimuth on 8/10 seeds (the
# other two reached 0.66 and 0.87), so check task1_model_test_acc and
# task2_model_test_acc in each summary.json before reading repr_distance.
#
# Budget: 10 epochs for each from-scratch model and 10 for the head, batch 256
# (the settings the width sweep used). That is 30 epochs at ~11 s on an H100
# plus ~60 s to load the h5 and ~10 s for the zero-shot pass: ~7 min per pair,
# hence the 45 min wall. Peak RSS is ~13 GB as in the single-task runs: both
# tasks wrap the same in-memory split, nothing is copied per task.
#
# The h5 file must already be in the cache: GPU nodes have no outbound network.
#
# Usage:
#   cd plasticity && sbatch experiments/submit_repr_distance_array.sh
#   sbatch --array=3,7 experiments/submit_repr_distance_array.sh           # re-run two pairs
#   HEAD_EPOCHS=3 sbatch experiments/submit_repr_distance_array.sh         # cheaper probe
#   LABEL_RULE=parity sbatch experiments/submit_repr_distance_array.sh      # binary azimuth
#   TARGET_FACTOR=4 MODEL_KIND=mlp_2x100 sbatch experiments/submit_repr_distance_array.sh
#   N_SAMPLES=50000 EPOCHS=1 sbatch --array=0 --time=00:15:00 experiments/submit_repr_distance_array.sh

set -euo pipefail

# ── locate the project (the dir holding experiments/ data/ models/) ───────────
_find_root() {
    local d="$1"
    while [[ "$d" != "/" ]]; do
        [[ -f "$d/experiments/calculate_repr_distance.py" ]] && { echo "$d"; return 0; }
        d="$(dirname "$d")"
    done
    return 1
}
PROJECT_ROOT="${PROJECT_ROOT:-$(_find_root "${SLURM_SUBMIT_DIR:-$PWD}" \
    || _find_root "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")}" || {
    echo "ERROR: cannot find the project (no experiments/calculate_repr_distance.py" >&2
    echo "       above ${SLURM_SUBMIT_DIR:-$PWD}). Submit from the plasticity/ directory." >&2
    exit 1; }
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
[[ -x "$UV" ]] || { echo "ERROR: uv not found (set UV=/path/to/uv)" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/.venv" ]] || {
    echo "ERROR: no .venv in $PROJECT_ROOT -- run 'uv sync' from a login node first." >&2
    exit 1; }

H5="${PLASTICITY_DATA_DIR:-$PROJECT_ROOT/data/_cache}/shapes3d/3dshapes.h5"
[[ -f "$H5" ]] || {
    echo "ERROR: $H5 is missing and GPU nodes cannot download it." >&2
    echo "       From a login node: cd $PROJECT_ROOT && uv run python -c \\" >&2
    echo "         \"from data.download import download_shapes3d; download_shapes3d()\"" >&2
    exit 1; }

# ── sweep ─────────────────────────────────────────────────────────────────────
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
I="${SLURM_ARRAY_TASK_ID:-0}"
if (( I >= ${#SEEDS[@]} )); then
    echo "ERROR: array index $I outside the sweep (0-$(( ${#SEEDS[@]} - 1 )))" >&2
    exit 1
fi
SEED="${SEEDS[$I]}"

OPTIMIZER="${OPTIMIZER:-sgd}"           # plain SGD, momentum 0 (build_optimizer default)
# the optimizer is part of the output folder: the Adam runs of shapes3d_reprdist_perm100/ predate this
RUN_TAG="${RUN_TAG:-shapes3d_reprdist_perm100_$OPTIMIZER}"
MODEL_KIND="${MODEL_KIND:-mlp_2x30}"
TARGET_FACTOR="${TARGET_FACTOR:-5}"      # object_azimuth, 15 classes
LABEL_RULE="${LABEL_RULE:-}"             # "" = the factor itself, parity = even -> 1, odd -> 0
EPOCHS="${EPOCHS:-10}"
HEAD_EPOCHS="${HEAD_EPOCHS:-$EPOCHS}"
BATCH="${BATCH:-256}"
LR="${LR:-0.001}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
WORKERS=$(( WORKERS > 1 ? WORKERS - 1 : 0 ))
OUTDIR="${OUTDIR:-results/$RUN_TAG}"
mkdir -p "$OUTDIR"

echo "Array task $I -> pair seed=$SEED  model=$MODEL_KIND  target_factor=$TARGET_FACTOR" \
     " label_rule=${LABEL_RULE:-none}" \
     " epochs=$EPOCHS  head_epochs=$HEAD_EPOCHS  batch=$BATCH  optimizer=$OPTIMIZER lr=$LR"
echo "outdir=$OUTDIR (one reprdist_* directory per pair, stable across resubmissions)"

srun "$UV" run --no-sync python -u -m experiments.calculate_repr_distance \
    --dataset_kind shapes3d \
    --perm_percentage 1.0 \
    --target_factor "$TARGET_FACTOR" \
    ${LABEL_RULE:+--label_rule "$LABEL_RULE"} \
    ${N_SAMPLES:+--n_samples "$N_SAMPLES"} \
    --model_kind "$MODEL_KIND" \
    --optimizer "$OPTIMIZER" \
    --learning_rate "$LR" \
    --num_epochs "$EPOCHS" \
    --head_epochs "$HEAD_EPOCHS" \
    --batch_size "$BATCH" \
    --num_workers "$WORKERS" \
    --seed "$SEED" \
    --save_dir "$OUTDIR" \
    "$@"

echo "Finished. Outputs in $OUTDIR"
