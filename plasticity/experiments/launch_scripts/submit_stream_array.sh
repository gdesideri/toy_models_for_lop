#!/bin/bash -l
#SBATCH --job-name=stream
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append

# run_tasks_stream.py on the class-attribute or cluster-factor tasks, one array task per
# (item, seed). Submitted by launch_all.sh, which sets the variables and --array; it can
# also be submitted by hand:
#
#   MODE=class   CLASS_TASKS="animal four_legs fur wings wheels"   (tasks/class_attributes.py)
#   MODE=cluster CLUSTER_FACTORS=6 CLUSTERS=8 CLUSTER_DIMS=50       (data/cluster_factors.py)
#   SINGLE=0  one stream of N_TASKS tasks per seed: array index = seed index
#   SINGLE=1  a fresh network per task (the stream's tasks one at a time, 1 task each):
#             array index I -> item ITEMS[I / #seeds], seed SEEDS[I % #seeds]
#
#   e.g. MODE=class DATASET=cifar10 N_TASKS=600 EPOCHS=10 RUN_TAG=cifar10_class_stream \
#        sbatch --array=0-2 experiments/launch_scripts/submit_stream_array.sh
#
# Streams stop at DEADLINE_H and resubmit themselves with --resume while they make progress
# (as submit_tasks_stream_array.sh). DRY_RUN=1 prints the command instead of running it.

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
    echo "ERROR: cannot find the project; submit from the plasticity/ directory." >&2
    exit 1; }
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
if [[ -z "${DRY_RUN:-}" ]]; then
    [[ -x "$UV" ]] || { echo "ERROR: uv not found (set UV=/path/to/uv)" >&2; exit 1; }
    [[ -d "$PROJECT_ROOT/.venv" ]] || {
        echo "ERROR: no .venv in $PROJECT_ROOT -- run 'uv sync' from a login node first." >&2; exit 1; }
fi

# ── the dataset must be downloaded already (GPU nodes are offline) ────────────
DATASET="${DATASET:-cifar10}"
case "$DATASET" in
    mnist)    NEEDED="mnist/MNIST/raw" ;;
    cifar10)  NEEDED="cifar10/cifar-10-batches-py" ;;
    shapes3d) NEEDED="shapes3d/3dshapes.h5" ;;
    *) echo "ERROR: DATASET=$DATASET: use mnist, cifar10 or shapes3d" >&2; exit 1 ;;
esac
[[ -e "${PLASTICITY_DATA_DIR:-$PROJECT_ROOT/data/_cache}/$NEEDED" ]] || {
    echo "ERROR: $DATASET is not downloaded and GPU nodes cannot download it (see launch_all.sh)." >&2
    exit 1; }

# ── what to run ───────────────────────────────────────────────────────────────
MODE="${MODE:?set MODE=class or MODE=cluster}"
SINGLE="${SINGLE:-0}"
CLASS_TASKS="${CLASS_TASKS:-animal four_legs fur wings wheels}"
CLUSTER_FACTORS="${CLUSTER_FACTORS:-6}"
CLUSTERS="${CLUSTERS:-8}"
CLUSTER_DIMS="${CLUSTER_DIMS:-50}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
case "$MODE" in
    class)   read -r -a ITEMS <<< "$CLASS_TASKS" ;;
    cluster) read -r -a ITEMS <<< "$(seq -s ' ' 0 $(( CLUSTER_FACTORS - 1 )))" ;;
    *) echo "ERROR: MODE=$MODE: use class or cluster" >&2; exit 1 ;;
esac

I="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$SINGLE" == 1 ]]; then
    N_RUNS=$(( ${#ITEMS[@]} * ${#SEEDS[@]} ))
    ITEM="${ITEMS[$(( I / ${#SEEDS[@]} ))]:-}"
    SEED="${SEEDS[$(( I % ${#SEEDS[@]} ))]}"
else
    N_RUNS=${#SEEDS[@]}
    SEED="${SEEDS[$I]:-}"
fi
if (( I >= N_RUNS )); then
    echo "ERROR: array index $I outside the sweep (0-$(( N_RUNS - 1 )))" >&2; exit 1
fi

MODEL_KIND="${MODEL_KIND:-mlp_2x30}"
OPTIMIZER="${OPTIMIZER:-sgd}"
LR="${LR:-0.001}"
BATCH="${BATCH:-256}"
EPOCHS="${EPOCHS:-10}"
N_TASKS="${N_TASKS:-600}"
PER_LABEL="${PER_LABEL:-10000}"
N_SAMPLES="${N_SAMPLES:-}"
DEADLINE_H="${DEADLINE_H:-23.3}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
WORKERS=$(( WORKERS > 1 ? WORKERS - 1 : 0 ))
RUN_TAG="${RUN_TAG:-${DATASET}_${MODE}_$([[ "$SINGLE" == 1 ]] && echo single || echo stream)_${MODEL_KIND}}"
OUTDIR="${OUTDIR:-results/$RUN_TAG}"

# task source and the run-directory suffix run_dir_for gives it (to count finished tasks)
if [[ "$MODE" == class ]]; then
    TASKS_NOW="${ITEM:-$CLASS_TASKS}"
    TASK_ARGS=(--class_tasks $TASKS_NOW --per_label "$PER_LABEL")
    SUFFIX="attrs-${TASKS_NOW// /-}"
    if [[ "$PER_LABEL" != 10000 ]]; then SUFFIX+="_n$PER_LABEL"; fi
else
    TASK_ARGS=(--cluster_factors "$CLUSTER_FACTORS" --clusters "$CLUSTERS" --cluster_dims "$CLUSTER_DIMS")
    SUFFIX="clu${CLUSTER_FACTORS}x${CLUSTERS}_d${CLUSTER_DIMS}"
    if [[ "$SINGLE" == 1 ]]; then
        TASK_ARGS+=(--task_factors "$ITEM")
        SUFFIX+="_features$ITEM"
    fi
fi
[[ "$SINGLE" == 1 ]] && N_TASKS=1
[[ -n "$N_SAMPLES" ]] && TASK_ARGS+=(--n_samples "$N_SAMPLES")

CMD=("$UV" run --no-sync python -u -m experiments.run_tasks_stream
     --dataset_kind "$DATASET" --perm_percentage 0 "${TASK_ARGS[@]}"
     --model_kind "$MODEL_KIND" --optimizer "$OPTIMIZER" --learning_rate "$LR"
     --num_epochs "$EPOCHS" --batch_size "$BATCH" --n_tasks "$N_TASKS"
     --snapshot_every "$EPOCHS" --num_workers "$WORKERS" --seed "$SEED"
     --save_dir "$OUTDIR" --resume --deadline_h "$DEADLINE_H" "$@")
if [[ -n "${DRY_RUN:-}" ]]; then
    echo "${CMD[*]}"
    exit 0
fi
mkdir -p "$OUTDIR"

# tasks already completed for this run (0 for a fresh run)
_done() {
    local s
    s=$(ls "$OUTDIR"/stream_*_"$SUFFIX"_seed"$SEED"/summary.json 2>/dev/null | head -1)
    [[ -n "$s" ]] && "$UV" run --no-sync python -c \
        "import json,sys; print(len(json.load(open(sys.argv[1]))['tasks']))" "$s" || echo 0
}
DONE_BEFORE=$(_done)
echo "array task $I -> mode=$MODE single=$SINGLE item=${ITEM:-all} seed=$SEED dataset=$DATASET" \
     "model=$MODEL_KIND tasks=$N_TASKS (done: $DONE_BEFORE) epochs=$EPOCHS lr=$LR batch=$BATCH -> $OUTDIR"

srun "${CMD[@]}"

DONE_AFTER=$(_done)
echo "seed $SEED: $DONE_AFTER/$N_TASKS tasks done (this job: +$(( DONE_AFTER - DONE_BEFORE )))"
if (( DONE_AFTER < N_TASKS )); then
    if (( DONE_AFTER <= DONE_BEFORE )); then
        echo "ERROR: no progress in this job -- not resubmitting. Check the log." >&2
        exit 1
    fi
    SELF="$PROJECT_ROOT/experiments/launch_scripts/submit_stream_array.sh"
    echo "resubmitting index $I: $(sbatch --parsable --array="$I" \
        --job-name="${SLURM_JOB_NAME:-stream}" "$SELF" "$@")"
fi
echo "Finished. Outputs in $OUTDIR"
