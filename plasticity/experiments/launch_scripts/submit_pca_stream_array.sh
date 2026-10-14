#!/bin/bash -l
#SBATCH --job-name=pca-stream
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append

# run_tasks_stream.py on PCA factors (data/pca_factors.py): the first PCA_COMPONENTS
# principal components of the raw train pixels of DATASET, each cut into K equal-count
# bins; task n is the parity of the bin on component n % PCA_COMPONENTS. One network
# carries its weights through the stream. Plain SGD, momentum 0, no pixel permutation.
#
# The sweep is K x seed, one array task per pair: index I -> K = BINS[I / #seeds],
# seed = SEEDS[I % #seeds]. Default 4 bin counts x 3 seeds = array 0-11. K = 2 is the
# median split (linearly separable); larger K is increasingly nonlinear.
#
# The PCA bins are cached in data/_cache/pca/ by the first job of each K that gets
# there (written atomically, so jobs starting together cannot corrupt it).
#
# Long streams stop at DEADLINE_H and resubmit themselves with --resume, as in
# submit_tasks_stream_array.sh; only if the job made progress.
#
# Usage:
#   cd plasticity && sbatch experiments/launch_scripts/submit_pca_stream_array.sh
#   DATASET=cifar10 sbatch experiments/launch_scripts/submit_pca_stream_array.sh
#   BINS="8" SEEDS="0" sbatch --array=0 experiments/launch_scripts/submit_pca_stream_array.sh   # one run
#   N_TASKS=6 EPOCHS=2 sbatch --time=01:00:00 experiments/launch_scripts/submit_pca_stream_array.sh

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

# ── the dataset must be downloaded already (GPU nodes are offline) ────────────
DATASET="${DATASET:-mnist}"
case "$DATASET" in
    mnist)   RAW="mnist/MNIST/raw" ;;
    cifar10) RAW="cifar10/cifar-10-batches-py" ;;
    *) echo "ERROR: DATASET=$DATASET: use mnist or cifar10 (fixed splits, so the PCA bins are cached)" >&2
       exit 1 ;;
esac
[[ -d "${PLASTICITY_DATA_DIR:-$PROJECT_ROOT/data/_cache}/$RAW" ]] || {
    echo "ERROR: $DATASET is not downloaded and GPU nodes cannot download it." >&2
    echo "       From a login node: cd $PROJECT_ROOT && uv run python -c \\" >&2
    echo "         \"from data.download import get_dataset; get_dataset('$DATASET')\"" >&2
    exit 1; }

# ── sweep ─────────────────────────────────────────────────────────────────────
read -r -a BINS_LIST <<< "${BINS:-2 4 8 16}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
I="${SLURM_ARRAY_TASK_ID:-0}"
N_RUNS=$(( ${#BINS_LIST[@]} * ${#SEEDS[@]} ))
if (( I >= N_RUNS )); then
    echo "ERROR: array index $I outside the sweep (0-$(( N_RUNS - 1 )))" >&2
    exit 1
fi
K="${BINS_LIST[$(( I / ${#SEEDS[@]} ))]}"
SEED="${SEEDS[$(( I % ${#SEEDS[@]} ))]}"

OPTIMIZER="${OPTIMIZER:-sgd}"            # plain SGD, momentum 0 (build_optimizer default)
PCA_COMPONENTS="${PCA_COMPONENTS:-6}"
RUN_TAG="${RUN_TAG:-${DATASET}_pca_stream_$OPTIMIZER}"
MODEL_KIND="${MODEL_KIND:-mlp_2x30}"
N_TASKS="${N_TASKS:-60}"                 # 10 passes over the 6 components
EPOCHS="${EPOCHS:-10}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-5}"
BATCH="${BATCH:-256}"
LR="${LR:-0.001}"
DEADLINE_H="${DEADLINE_H:-23.3}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
WORKERS=$(( WORKERS > 1 ? WORKERS - 1 : 0 ))
OUTDIR="${OUTDIR:-results/$RUN_TAG}"
mkdir -p "$OUTDIR"

# tasks already completed for this (K, seed) (0 for a fresh run)
_done() {
    local s
    s=$(ls "$OUTDIR"/stream_*_pca"$PCA_COMPONENTS"_bins"$K"_seed"$SEED"/summary.json 2>/dev/null | head -1)
    [[ -n "$s" ]] && "$UV" run --no-sync python -c \
        "import json,sys; print(len(json.load(open(sys.argv[1]))['tasks']))" "$s" || echo 0
}
DONE_BEFORE=$(_done)

echo "Array task $I -> K=$K seed=$SEED  dataset=$DATASET  components=$PCA_COMPONENTS  model=$MODEL_KIND" \
     " tasks=$N_TASKS (done: $DONE_BEFORE)  epochs=$EPOCHS  batch=$BATCH  optimizer=$OPTIMIZER lr=$LR" \
     " snapshot_every=$SNAPSHOT_EVERY"
echo "outdir=$OUTDIR"

srun "$UV" run --no-sync python -u -m experiments.run_tasks_stream \
    --dataset_kind "$DATASET" \
    --perm_percentage 0 \
    --pca_components "$PCA_COMPONENTS" \
    --pca_bins "$K" \
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
echo "K=$K seed $SEED: $DONE_AFTER/$N_TASKS tasks done (this job: +$(( DONE_AFTER - DONE_BEFORE )))"
if (( DONE_AFTER < N_TASKS )); then
    if (( DONE_AFTER <= DONE_BEFORE )); then
        echo "ERROR: no progress in this job -- not resubmitting. Check the log." >&2
        exit 1
    fi
    SELF="$PROJECT_ROOT/experiments/launch_scripts/submit_pca_stream_array.sh"
    echo "resubmitting K=$K seed $SEED to continue: $(cd "$PROJECT_ROOT" && sbatch --parsable --array="$I" \
        --job-name="${SLURM_JOB_NAME:-pca-stream}" "$SELF" "$@")"
fi
echo "Finished. Outputs in $OUTDIR"
