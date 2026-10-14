#!/bin/bash -l
#SBATCH --job-name=s3d-stream
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append

# run_tasks_stream.py on Shapes3D, permutation family: one mlp_2x30 trained on 800
# consecutive tasks, each a fresh 100% pixel permutation, each starting from the
# weights the previous task ended with. Plain SGD, momentum 0.
#
# One array task = one seed; the 3 seeds run in parallel. The seed draws the
# train/test split, the 800 permutations and the init.
#
# Label: object_azimuth parity (even bin -> 1, odd -> 0), as in the parity sweeps;
# LABEL_RULE="" gives the 15-class label. Budget per task as in every other
# experiment: 10 epochs, batch 256, lr 0.001. A fresh mlp_2x30 solves this label in
# 2-3 epochs with SGD (learning-rate probe, jobs 888528-888542).
#
# Time: ~9.4 s/epoch on an H100 -> ~100 s per task -> ~22 h for 800 tasks, right at
# the 24 h wall. So the stream stops itself between tasks at DEADLINE_H (23.3 h) and
# this script resubmits the same seed with --resume, which continues exactly where
# it stopped (weights, RNG state and task draws are restored). It resubmits only if
# the job made progress, so a crashing run cannot loop.
#
# Disk: a snapshot is 1.5 MB. Every epoch would be 800 x 10 x 1.5 MB = 12 GB per
# seed, more than the home quota has left for three seeds, so the default is every
# 5 epochs (mid-task and end of task): ~2.4 GB + ~0.3 GB of histories per seed.
#
# Feature-switch family: PERM=0 TASK_FACTORS="0 1 2 3 4 5" N_TASKS=6 -- no pixel shuffle,
# task n is the parity of Shapes3D factor n (floor, wall, object hue, size, type, azimuth).
#
# Usage:
#   cd plasticity && sbatch experiments/submit_tasks_stream_array.sh
#   sbatch --array=1 experiments/submit_tasks_stream_array.sh          # (re)start/continue one seed
#   LABEL_RULE= sbatch experiments/submit_tasks_stream_array.sh        # 15-class azimuth
#   N_TASKS=50 EPOCHS=2 sbatch --time=01:00:00 experiments/submit_tasks_stream_array.sh   # short probe
#   PERM=0 TASK_FACTORS="0 1 2 3 4 5" N_TASKS=6 RUN_TAG=shapes3d_stream_features_sgd \
#       sbatch --time=00:45:00 experiments/submit_tasks_stream_array.sh             # feature switches
#   PERM=0 TASK_FACTORS="0 1 2 3 4 5" FEATURES_PER_TASK=2 RUN_TAG=shapes3d_stream_feature_pairs_sgd \
#       sbatch experiments/submit_tasks_stream_array.sh                              # two-feature swaps
#   PERM=0 TASK_FACTORS="0 1 2 3 4 5" FEATURES_PER_TASK=3 KEEP_FEATURES=1 RUN_TAG=shapes3d_stream_feature_triples_sgd \
#       sbatch experiments/submit_tasks_stream_array.sh                              # three features, keep 1

set -euo pipefail

# ── locate the project (the dir holding experiments/ data/ models/) ───────────
_find_root() {
    local d="$1"
    while [[ "$d" != "/" ]]; do
        [[ -f "$d/experiments/run_tasks_stream.py" ]] && { echo "$d"; return 0; }
        d="$(dirname "$d")"
    done
    return 1
}
PROJECT_ROOT="${PROJECT_ROOT:-$(_find_root "${SLURM_SUBMIT_DIR:-$PWD}" \
    || _find_root "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")}" || {
    echo "ERROR: cannot find the project (no experiments/run_tasks_stream.py" >&2
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
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
I="${SLURM_ARRAY_TASK_ID:-0}"
if (( I >= ${#SEEDS[@]} )); then
    echo "ERROR: array index $I outside the sweep (0-$(( ${#SEEDS[@]} - 1 )))" >&2
    exit 1
fi
SEED="${SEEDS[$I]}"

OPTIMIZER="${OPTIMIZER:-sgd}"            # plain SGD, momentum 0 (build_optimizer default)
RUN_TAG="${RUN_TAG:-shapes3d_stream_perm100_$OPTIMIZER}"
MODEL_KIND="${MODEL_KIND:-mlp_2x30}"
TARGET_FACTOR="${TARGET_FACTOR:-5}"      # object_azimuth
PERM="${PERM:-1.0}"                      # fraction of pixels permuted per task
TASK_FACTORS="${TASK_FACTORS:-}"         # e.g. "0 1 2 3 4 5": task n labelled by factor n
FEATURES_PER_TASK="${FEATURES_PER_TASK:-1}"  # 2: task n = parity of the pair (factor n, factor n+1)
KEEP_FEATURES="${KEEP_FEATURES:-}"       # with 3 features per task: features kept per switch (1 or 2)
LABEL_RULE="${LABEL_RULE-parity}"        # "" = the 15 azimuth classes
N_TASKS="${N_TASKS:-800}"
EPOCHS="${EPOCHS:-10}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-5}"
BATCH="${BATCH:-256}"
LR="${LR:-0.001}"
DEADLINE_H="${DEADLINE_H:-23.3}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
WORKERS=$(( WORKERS > 1 ? WORKERS - 1 : 0 ))
OUTDIR="${OUTDIR:-results/$RUN_TAG}"
mkdir -p "$OUTDIR"

# tasks already completed for this seed (0 for a fresh run)
_done() {
    local s
    s=$(ls "$OUTDIR"/stream_*_seed"$SEED"/summary.json 2>/dev/null | head -1)
    [[ -n "$s" ]] && "$UV" run --no-sync python -c \
        "import json,sys; print(len(json.load(open(sys.argv[1]))['tasks']))" "$s" || echo 0
}
DONE_BEFORE=$(_done)

echo "Array task $I -> seed=$SEED  model=$MODEL_KIND  target_factor=$TARGET_FACTOR" \
     " label_rule=${LABEL_RULE:-none}  perm=$PERM  task_factors=${TASK_FACTORS:-none}  features_per_task=$FEATURES_PER_TASK  keep=${KEEP_FEATURES:-default}  tasks=$N_TASKS (done: $DONE_BEFORE)  epochs=$EPOCHS" \
     " batch=$BATCH  optimizer=$OPTIMIZER lr=$LR  snapshot_every=$SNAPSHOT_EVERY"
echo "outdir=$OUTDIR"

srun "$UV" run --no-sync python -u -m experiments.run_tasks_stream \
    --dataset_kind shapes3d \
    --perm_percentage "$PERM" \
    --target_factor "$TARGET_FACTOR" \
    ${TASK_FACTORS:+--task_factors $TASK_FACTORS} \
    ${TASK_FACTORS:+--features_per_task "$FEATURES_PER_TASK"} \
    ${KEEP_FEATURES:+--keep_features "$KEEP_FEATURES"} \
    ${LABEL_RULE:+--label_rule "$LABEL_RULE"} \
    ${N_SAMPLES:+--n_samples "$N_SAMPLES"} \
    --model_kind "$MODEL_KIND" \
    --optimizer "$OPTIMIZER" \
    --learning_rate "$LR" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --n_tasks "$N_TASKS" \
    --snapshot_every "$SNAPSHOT_EVERY" \
    --num_workers "$WORKERS" \
    --seed "$SEED" \
    --save_dir "$OUTDIR" \
    --resume \
    --deadline_h "$DEADLINE_H" \
    "$@"

DONE_AFTER=$(_done)
echo "seed $SEED: $DONE_AFTER/$N_TASKS tasks done (this job: +$(( DONE_AFTER - DONE_BEFORE )))"
if (( DONE_AFTER < N_TASKS )); then
    if (( DONE_AFTER <= DONE_BEFORE )); then
        echo "ERROR: no progress in this job -- not resubmitting. Check the log." >&2
        exit 1
    fi
    SELF="$PROJECT_ROOT/experiments/launch_scripts/submit_tasks_stream_array.sh"
    echo "resubmitting seed $SEED to continue: $(cd "$PROJECT_ROOT" && sbatch --parsable --array="$I" \
        --job-name="${SLURM_JOB_NAME:-s3d-stream}" "$SELF" "$@")"
fi
echo "Finished. Outputs in $OUTDIR"
