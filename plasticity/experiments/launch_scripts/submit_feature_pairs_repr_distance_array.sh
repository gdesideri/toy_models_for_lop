#!/bin/bash -l
#SBATCH --job-name=s3d-rd-feat
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-299
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append

# calculate_repr_distance.py on Shapes3D for pairs of tasks that differ in the
# FEATURE they classify, not in a pixel permutation: pixels are left unshuffled
# (--perm_percentage 0), task 1 is binary parity on feature A, task 2 binary
# parity on feature B. Same evaluation as the permuted-pixel sweep:
#   1. train a model on task 1 and a model on task 2, from scratch
#   2. zero-shot: loss/acc of the task-1 model on task 2's training set
#   3. reset the task-1 model's head, freeze its body, retrain the head on task 2
#   4. repr distance = test acc(task-2 model) - test acc(task-1 body + new head)
#
# Labels: parity of the feature's value index -- even -> class 1, odd -> class 0:
#   0 floor_hue (10 values)   1 wall_hue (10)   2 object_hue (10)
#   3 object_size (8)         4 object_type (4: cube,sphere -> 1; cylinder,capsule -> 0)
#   5 object_azimuth (15: 8 even bins vs 7 odd)
# Model: mlp_2x30, 10 epochs per model and for the head, batch 256.
#
# The sweep: FACTOR_PAIRS x SEEDS. Array index I runs pair I / N_SEEDS with seed
# I % N_SEEDS. The default is every ORDERED pair of distinct features (30 pairs:
# the distance is asymmetric -- task 1's body is scored on task 2) x 10 seeds =
# 300 array tasks. The seed draws the train/test split and both inits, so seed k
# uses the same split for every pair.
#
# Cost: ~7 min per task (30 epochs at ~11 s + ~60 s h5 load), so the full default
# sweep is ~35 GPU-hours. For a first look run one pair, e.g. azimuth -> type:
#   FACTOR_PAIRS="5:4" sbatch --array=0-9 experiments/submit_feature_pairs_repr_distance_array.sh
# --array must cover N_PAIRS x N_SEEDS - 1; an index past the sweep exits with an error.
#
# Caveat from the permuted-pixel parity sweep (job 887812): at this width and
# budget 8/20 from-scratch parity models did not solve azimuth parity. Check
# task1_model_test_acc and task2_model_test_acc in each summary.json before
# reading repr_distance.
#
# The h5 file must already be in the cache: GPU nodes have no outbound network.
#
# Usage:
#   cd plasticity && sbatch experiments/submit_feature_pairs_repr_distance_array.sh
#   FACTOR_PAIRS="5:4 4:5" sbatch --array=0-19 experiments/submit_feature_pairs_repr_distance_array.sh
#   FACTOR_PAIRS="5:4" SEEDS="3 7" sbatch --array=0-1 experiments/submit_feature_pairs_repr_distance_array.sh

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
ALL_PAIRS=""
for a in 0 1 2 3 4 5; do for b in 0 1 2 3 4 5; do
    [[ $a -ne $b ]] && ALL_PAIRS+="$a:$b "
done; done
read -r -a PAIRS <<< "${FACTOR_PAIRS:-$ALL_PAIRS}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
for p in "${PAIRS[@]}"; do
    [[ "$p" =~ ^[0-5]:[0-5]$ && "${p%:*}" != "${p#*:}" ]] || {
        echo "ERROR: bad pair '$p' in FACTOR_PAIRS (want A:B, factors 0-5, A != B)" >&2; exit 1; }
done

N_SEEDS=${#SEEDS[@]}
TOTAL=$(( ${#PAIRS[@]} * N_SEEDS ))
I="${SLURM_ARRAY_TASK_ID:-0}"
if (( I >= TOTAL )); then
    echo "ERROR: array index $I outside the sweep (0-$(( TOTAL - 1 )):" \
         "${#PAIRS[@]} pairs x $N_SEEDS seeds)" >&2
    exit 1
fi
PAIR="${PAIRS[$(( I / N_SEEDS ))]}"
SEED="${SEEDS[$(( I % N_SEEDS ))]}"
FACTOR1="${PAIR%:*}"
FACTOR2="${PAIR#*:}"

OPTIMIZER="${OPTIMIZER:-sgd}"           # plain SGD, momentum 0 (build_optimizer default)
# the optimizer is part of the output folder: the Adam runs of shapes3d_reprdist_features/ predate this
RUN_TAG="${RUN_TAG:-shapes3d_reprdist_features_$OPTIMIZER}"
MODEL_KIND="${MODEL_KIND:-mlp_2x30}"
LABEL_RULE="${LABEL_RULE:-parity}"
EPOCHS="${EPOCHS:-10}"
HEAD_EPOCHS="${HEAD_EPOCHS:-$EPOCHS}"
BATCH="${BATCH:-256}"
LR="${LR:-0.001}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
WORKERS=$(( WORKERS > 1 ? WORKERS - 1 : 0 ))
OUTDIR="${OUTDIR:-results/$RUN_TAG}"
mkdir -p "$OUTDIR"

echo "Array task $I -> factors $FACTOR1 -> $FACTOR2 ($LABEL_RULE)  seed=$SEED  model=$MODEL_KIND" \
     " epochs=$EPOCHS  head_epochs=$HEAD_EPOCHS  batch=$BATCH  optimizer=$OPTIMIZER lr=$LR"
echo "outdir=$OUTDIR (one reprdist_* directory per pair and seed, stable across resubmissions)"

srun "$UV" run --no-sync python -u -m experiments.calculate_repr_distance \
    --dataset_kind shapes3d \
    --perm_percentage 0 \
    --target_factor "$FACTOR1" \
    --task2_target_factor "$FACTOR2" \
    --label_rule "$LABEL_RULE" \
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
