#!/bin/bash -l
#SBATCH --job-name=s3d-feat-width
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append

# Width sweep of the feature-switch stream (run_tasks_stream.py): one MLP trained on
# 800 consecutive Shapes3D tasks, each starting from the weights the previous task
# ended with. No pixel shuffle; task n is the parity of Shapes3D factor n mod 6
# (floor hue, wall hue, object hue, size, type, azimuth): label 1 if the factor's
# value index is even. Same settings as shapes3d_stream_features_sgd (mlp_2x30):
# 10 epochs per task, batch 256, plain SGD (momentum 0), lr 0.001, snapshots every
# 5 epochs.
#
# Sweep: widths 60 90 120 150 180 210 x seeds 0 1 2 = 18 array tasks;
# array index i -> width WIDTHS[i / 3], seed SEEDS[i % 3].
#
# Time: ~22 h per run (data-bound, so roughly width-independent). The stream stops
# itself between tasks at DEADLINE_H (23.3 h) and this script resubmits the same
# array index with --resume, only if the job made progress.
#
# Disk: a snapshot is ~4 B x W(12292 + W): 3.0 MB (2x60) to 10.5 MB (2x210), so the
# whole sweep is ~195 GB -- far beyond the home quota. Results go to $WORK and
# results/$RUN_TAG is a symlink to it.
#
# Usage:
#   cd plasticity && sbatch experiments/launch_scripts/submit_feature_stream_width_array.sh
#   sbatch --array=15 experiments/launch_scripts/submit_feature_stream_width_array.sh   # (re)start one run
#   N_TASKS=3 sbatch --array=15 --time=00:30:00 experiments/launch_scripts/submit_feature_stream_width_array.sh  # probe

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
    exit 1; }

# ── sweep ─────────────────────────────────────────────────────────────────────
read -r -a WIDTHS <<< "${WIDTHS:-60 90 120 150 180 210}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
I="${SLURM_ARRAY_TASK_ID:-0}"
if (( I >= ${#WIDTHS[@]} * ${#SEEDS[@]} )); then
    echo "ERROR: array index $I outside the sweep (0-$(( ${#WIDTHS[@]} * ${#SEEDS[@]} - 1 )))" >&2
    exit 1
fi
WIDTH="${WIDTHS[$(( I / ${#SEEDS[@]} ))]}"
SEED="${SEEDS[$(( I % ${#SEEDS[@]} ))]}"
MODEL_KIND="mlp_2x$WIDTH"

OPTIMIZER="${OPTIMIZER:-sgd}"            # plain SGD, momentum 0 (build_optimizer default)
RUN_TAG="${RUN_TAG:-shapes3d_stream_features_width_$OPTIMIZER}"
TASK_FACTORS="${TASK_FACTORS:-0 1 2 3 4 5}"
N_TASKS="${N_TASKS:-800}"
EPOCHS="${EPOCHS:-10}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-5}"
BATCH="${BATCH:-256}"
LR="${LR:-0.001}"
DEADLINE_H="${DEADLINE_H:-23.3}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
WORKERS=$(( WORKERS > 1 ? WORKERS - 1 : 0 ))

# outputs live in an hnvme workspace; results/$RUN_TAG links there. Helma compute
# nodes mount only /home/hpc and /hnvme (no atuin, woody, vault, anvme). Workspaces expire (90 d,
# extend with `ws_extend -F hnvme plasticity_results 90`) -- copy results off before.
WS_DIR="${WS_DIR:-/hnvme/workspace/$USER-plasticity_results}"
[[ -d "$WS_DIR" ]] || {
    echo "ERROR: workspace $WS_DIR missing -- run 'ws_allocate -F hnvme plasticity_results 90' on a login node." >&2
    exit 1; }
STORE="${STORE:-$WS_DIR/$RUN_TAG}"
OUTDIR="results/$RUN_TAG"
mkdir -p "$STORE"
[[ -e "$OUTDIR" ]] || ln -s "$STORE" "$OUTDIR"

# tasks already completed for this width and seed (0 for a fresh run)
_done() {
    local s
    s=$(ls "$OUTDIR"/stream_*_"$MODEL_KIND"_*_seed"$SEED"/summary.json 2>/dev/null | head -1)
    [[ -n "$s" ]] && "$UV" run --no-sync python -c \
        "import json,sys; print(len(json.load(open(sys.argv[1]))['tasks']))" "$s" || echo 0
}
DONE_BEFORE=$(_done)

echo "Array task $I -> model=$MODEL_KIND seed=$SEED  task_factors=$TASK_FACTORS" \
     " tasks=$N_TASKS (done: $DONE_BEFORE)  epochs=$EPOCHS  batch=$BATCH  optimizer=$OPTIMIZER lr=$LR" \
     " snapshot_every=$SNAPSHOT_EVERY"
echo "outdir=$OUTDIR -> $STORE"

srun "$UV" run --no-sync python -u -m experiments.run_tasks_stream \
    --dataset_kind shapes3d \
    --perm_percentage 0 \
    --target_factor 5 \
    --task_factors $TASK_FACTORS \
    --features_per_task 1 \
    --label_rule parity \
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
echo "$MODEL_KIND seed $SEED: $DONE_AFTER/$N_TASKS tasks done (this job: +$(( DONE_AFTER - DONE_BEFORE )))"
if (( DONE_AFTER < N_TASKS )); then
    if (( DONE_AFTER <= DONE_BEFORE )); then
        echo "ERROR: no progress in this job -- not resubmitting. Check the log." >&2
        exit 1
    fi
    SELF="$PROJECT_ROOT/experiments/launch_scripts/submit_feature_stream_width_array.sh"
    echo "resubmitting array index $I to continue: $(cd "$PROJECT_ROOT" && sbatch --parsable --array="$I" \
        --job-name="${SLURM_JOB_NAME:-s3d-feat-width}" "$SELF" "$@")"
fi
echo "Finished. Outputs in $OUTDIR"
