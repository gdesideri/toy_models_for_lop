#!/bin/bash
# MNIST cluster-factor experiment, the counterpart of the CIFAR-10 cluster runs in launch_all.sh:
# plain SGD (lr 0.001, batch 256), 100 epochs per task, through submit_stream_array.sh.
#
#   mnist_cluster_stream   6 cluster factors x 8 clusters, 600 tasks x 100 epochs   (1 job per seed)
#   mnist_cluster_single   each cluster factor alone on a fresh network, 100 epochs  (6 jobs per seed)
#
# Run from a login node: it computes and caches MNIST's cluster factors first if they are
# missing (GPU nodes are offline, and the parallel jobs should not all fit them at once).
#
# Usage (from plasticity/):
#   bash experiments/launch_scripts/launch_mnist_cluster.sh                     # seeds 0 1 2, mlp_2x120
#   SEEDS="3 4 5" bash experiments/launch_scripts/launch_mnist_cluster.sh       # other seeds
#   MODELS="mlp_2x120 mlp_2x30" bash experiments/launch_scripts/launch_mnist_cluster.sh   # width control
#   ONLY=mnist_cluster_stream bash experiments/launch_scripts/launch_mnist_cluster.sh     # stream only
#   DRY_RUN=1 bash experiments/launch_scripts/launch_mnist_cluster.sh          # print the sbatch commands only

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."           # plasticity/
mkdir -p logs
SCRIPT=experiments/launch_scripts/submit_stream_array.sh
SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-mlp_2x120}"
ONLY="${ONLY:-mnist_cluster_stream mnist_cluster_single}"
FACTORS=6 CLUSTERS=8 DIMS=50
read -r -a SEED_LIST <<< "$SEEDS"
N_SEEDS=${#SEED_LIST[@]}
COMMON=(OPTIMIZER=sgd LR=0.001 BATCH=256 SEEDS="$SEEDS" MODE=cluster DATASET=mnist EPOCHS=100
        CLUSTER_FACTORS=$FACTORS CLUSTERS=$CLUSTERS CLUSTER_DIMS=$DIMS)

# the cluster factors, computed once on the login node
if [[ -z "${DRY_RUN:-}" ]]; then
    uv run python -c "
from data.download import get_dataset
from data.cluster_factors import with_cluster_factors, cluster_cache_path
p = cluster_cache_path('mnist', $FACTORS, $CLUSTERS, $DIMS)
print('cluster factors:', 'cached' if p.exists() else 'computing', p)
with_cluster_factors(*get_dataset('mnist'), $FACTORS, $CLUSTERS, $DIMS, cache_path=p)"
fi

submit() {   # submit <name> <array range> <time> <model> VAR=value ...
    local name="$1" array="$2" time="$3" model="$4"; shift 4
    [[ " $ONLY " == *" $name "* ]] || return 0
    local cmd=(env "${COMMON[@]}" MODEL_KIND="$model" RUN_TAG="${name}_${model}" "$@"
               sbatch --job-name="$name" --array="$array" --time="$time" "$SCRIPT")
    if [[ -n "${DRY_RUN:-}" ]]; then
        printf '%q ' "${cmd[@]}"; echo
    else
        echo "$name ($model, seeds $SEEDS): $("${cmd[@]}")"
    fi
}

for model in $MODELS; do
    submit mnist_cluster_stream 0-$(( N_SEEDS - 1 ))            24:00:00 "$model" N_TASKS=600
    submit mnist_cluster_single 0-$(( FACTORS * N_SEEDS - 1 ))  02:00:00 "$model" SINGLE=1
done
