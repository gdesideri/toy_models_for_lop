#!/bin/bash
# Submits every class-attribute / cluster-factor experiment: an mlp_2x30, plain SGD
# (lr 0.001, batch 256), seeds 0 1 2, through submit_stream_array.sh.
#
#   cifar10_class_stream     5 semantic attributes (animal four_legs fur wings wheels), 600 tasks x 10 epochs
#   cifar10_class_single     each attribute alone on a fresh network, 10 epochs           (5 x 3 jobs)
#   cifar10_cluster_stream   6 cluster factors x 8 clusters, 600 tasks x 40 epochs
#   cifar10_cluster_single   each cluster factor alone on a fresh network, 40 epochs       (6 x 3 jobs)
#   shapes3d_cluster_stream  control: cluster factors on 50k Shapes3D images, 600 tasks x 40 epochs
#
# Before the first launch, from a login node (cd plasticity):
#   uv sync
#   uv run python -c "from data.download import get_dataset; get_dataset('cifar10')"
#   uv run python -c "from data.download import download_shapes3d; download_shapes3d()"
#   uv run python -c "from data.download import get_dataset; from data.cluster_factors import with_cluster_factors, cluster_cache_path; with_cluster_factors(*get_dataset('cifar10'), 6, 8, 50, cache_path=cluster_cache_path('cifar10', 6, 8, 50))"
#
# Usage (from plasticity/):
#   bash experiments/launch_scripts/launch_all.sh                       # everything
#   ONLY="cifar10_class_stream cifar10_class_single" bash experiments/launch_scripts/launch_all.sh
#   DRY_RUN=1 bash experiments/launch_scripts/launch_all.sh             # print the sbatch commands only

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."           # plasticity/
mkdir -p logs
SCRIPT=experiments/launch_scripts/submit_stream_array.sh
ONLY="${ONLY:-cifar10_class_stream cifar10_class_single cifar10_cluster_stream cifar10_cluster_single shapes3d_cluster_stream}"
COMMON=(MODEL_KIND=mlp_2x30 OPTIMIZER=sgd LR=0.001 BATCH=256 SEEDS="0 1 2")

submit() {   # submit <name> <array range> <time> VAR=value ...
    local name="$1" array="$2" time="$3"; shift 3
    [[ " $ONLY " == *" $name "* ]] || return 0
    local cmd=(env "${COMMON[@]}" RUN_TAG="${name}_mlp_2x30" "$@"
               sbatch --job-name="$name" --array="$array" --time="$time" "$SCRIPT")
    if [[ -n "${DRY_RUN:-}" ]]; then
        printf '%q ' "${cmd[@]}"; echo
    else
        echo "$name: $("${cmd[@]}")"
    fi
}

submit cifar10_class_stream    0-2  24:00:00 MODE=class   DATASET=cifar10 N_TASKS=600 EPOCHS=10
submit cifar10_class_single    0-14 01:00:00 MODE=class   DATASET=cifar10 SINGLE=1    EPOCHS=10
submit cifar10_cluster_stream  0-2  24:00:00 MODE=cluster DATASET=cifar10 N_TASKS=600 EPOCHS=40
submit cifar10_cluster_single  0-17 02:00:00 MODE=cluster DATASET=cifar10 SINGLE=1    EPOCHS=40
submit shapes3d_cluster_stream 0-2  24:00:00 MODE=cluster DATASET=shapes3d N_SAMPLES=50000 N_TASKS=600 EPOCHS=40
