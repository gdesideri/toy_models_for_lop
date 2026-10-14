#!/bin/bash -l
#SBATCH --job-name=s3d-perm100
#SBATCH --account=v170be
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append

# run_basic_task_training.py on Shapes3D, 100% permuted pixels, 10 seeds.
#
# One array task = one seed = one independent replicate. The seed drives three
# things inside the script, in this order: the train/test split, the pixel
# permutation that defines the task, and the weight init. So the 10 runs are 10
# genuinely different permuted-100 tasks, not one task trained 10 times.
#
# Architecture: MLP, 10 units per layer -- mlp_2x10 on the 12 288-dim
# flattened image (123 044 parameters, nearly all of them in the first layer).
# Note that a fully connected first layer is invariant to a FIXED permutation of
# its inputs, so permuted-100 is as learnable for it as the unpermuted dataset --
# the permutation relabels which weight sees which pixel, nothing more. What
# varies across the 10 runs is the split, the permutation and the init, all from
# the seed.
#
# Epochs and batch size, from the dataset size:
#   Shapes3D is 480 000 images (64x64x3). With train_frac=0.9 that is
#   432 000 train / 48 000 test.
#   batch_size 256 -> 1688 optimizer steps per epoch. Small enough that an epoch
#   is still ~1700 updates rather than a few hundred, large enough that the
#   12 288 x 10 first layer is one efficient GEMM per step on an H100.
#   num_epochs 10  -> ~16 900 updates. The label here is object_type (4 classes,
#   balanced by construction, 120 000 images each). A width-10 net is the
#   bottleneck here rather than the step budget, so the later epochs measure
#   where it plateaus, which is the point of the sweep. At width 100 this label
#   saturated at 1.000 test accuracy on all 10 seeds (job 884791); width 10 is
#   the narrower net that sweep motivated.
#   Measured on an H100 (job 884789, 1-epoch probe at width 100): 11 s/epoch
#   plus ~60 s to read the h5 and build the split, so a 10-epoch run is
#   ~3.5 min; width 10 is no slower, the loader sets the pace.
#   Hence the 30 min wall -- generous, and it schedules faster than a long one.
#
# Depth is 2 hidden layers; MODEL_KIND=mlp_4x10 etc. changes it, any
# mlp_<n_layers>x<width> name resolves in models/factory.py.
#
# Memory: the images are loaded into RAM as uint8 (5.9 GB) and the split copies
# them once more; measured peak RSS is 13 GB. No --mem is requested -- the
# partition's default for one GPU (175 GB on the probe) is far more than that.
# Set N_SAMPLES to load a subset instead.
#
# The workers only stay that cheap because run_basic_task_training.py forces the
# "fork" start method. Python 3.14 defaults to forkserver, which pickles the
# whole 6 GB dataset into EVERY worker -- the first probe (job 884786) hit
# 43 GB RSS and was OOM-killed in two minutes.
#
# The h5 file must already be in the cache: GPU nodes have no outbound network.
# Download it once from a login node:
#   cd plasticity && uv run python -c \
#       "from data.download import download_shapes3d; download_shapes3d()"
#
# Usage:
#   cd plasticity && sbatch experiments/submit_shapes3d_perm100_array.sh
#   sbatch --array=3,7 experiments/submit_shapes3d_perm100_array.sh   # re-run two seeds
#   EPOCHS=20 BATCH=512 sbatch experiments/submit_shapes3d_perm100_array.sh
#   N_SAMPLES=50000 sbatch experiments/submit_shapes3d_perm100_array.sh   # quick probe

set -euo pipefail

# ── locate the project (the dir holding experiments/ data/ models/) ───────────
_find_root() {
    local d="$1"
    while [[ "$d" != "/" ]]; do
        [[ -f "$d/experiments/run_basic_task_training.py" ]] && { echo "$d"; return 0; }
        d="$(dirname "$d")"
    done
    return 1
}
PROJECT_ROOT="${PROJECT_ROOT:-$(_find_root "${SLURM_SUBMIT_DIR:-$PWD}" \
    || _find_root "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")}" || {
    echo "ERROR: cannot find the project (no experiments/run_basic_task_training.py" >&2
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
# the optimizer is part of the output folder: the Adam runs of shapes3d_perm100/ predate this
RUN_TAG="${RUN_TAG:-shapes3d_perm100_$OPTIMIZER}"
MODEL_KIND="${MODEL_KIND:-mlp_2x10}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-256}"
LR="${LR:-0.001}"
TARGET_FACTOR="${TARGET_FACTOR:-4}"      # object_type, 4 classes
LABEL_RULE="${LABEL_RULE:-}"             # "" = the factor itself, parity = even -> 1, odd -> 0
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
WORKERS=$(( WORKERS > 1 ? WORKERS - 1 : 0 ))
OUTDIR="${OUTDIR:-results/$RUN_TAG}"
mkdir -p "$OUTDIR"

echo "Array task $I -> seed=$SEED  model=$MODEL_KIND  epochs=$EPOCHS  batch=$BATCH  optimizer=$OPTIMIZER lr=$LR"
echo "outdir=$OUTDIR/shapes3d_${MODEL_KIND}_perm100_seed${SEED} (stable across resubmissions)"

srun "$UV" run --no-sync python -u -m experiments.run_basic_task_training \
    --dataset_kind shapes3d \
    --perm_percentage 1.0 \
    --target_factor "$TARGET_FACTOR" \
    ${LABEL_RULE:+--label_rule "$LABEL_RULE"} \
    ${N_SAMPLES:+--n_samples "$N_SAMPLES"} \
    --model_kind "$MODEL_KIND" \
    --optimizer "$OPTIMIZER" \
    --learning_rate "$LR" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --num_workers "$WORKERS" \
    --seed "$SEED" \
    --save_dir "$OUTDIR" \
    "$@"

echo "Finished. Outputs in $OUTDIR"
