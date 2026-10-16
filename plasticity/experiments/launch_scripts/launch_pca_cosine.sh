#!/bin/bash
# Submits the PCA-cosine benchmark (data/pca_factors.py): every train image is scored by the
# cosine similarity of the centred image (image - train mean) with each of the first 6 principal
# components of the train pixels, each score is cut into 10 equal-count bins, and task n is
# labelled by the bin on component n mod 6 with one of two rules, as separate experiments:
#   parity      even bin -> 1
#   upper_half  bin >= 5 -> 1 (the threshold at 0.5: score above the train median)
# Each gets a stream (600 tasks x 100 epochs, 3 seeds) and single-component baselines (a fresh
# network per component: 6 components x 3 seeds). Plain SGD lr 0.001, batch 256, unit health
# logged after every task. CIFAR-10 at mlp_2x120, MNIST at mlp_2x30.
#
# Before the first launch (login node, from plasticity/): datasets downloaded as in launch_all.sh;
# optionally build and inspect each benchmark, which also writes the cache the jobs read:
#   uv run python -m data.pca_factors --dataset cifar10 --components 6 --bins 10 --score cosine
#   uv run python -m data.pca_factors --dataset mnist --components 6 --bins 10 --score cosine
# Usage (from plasticity/):
#   DRY_RUN=1 bash experiments/launch_scripts/launch_pca_cosine.sh     # print the sbatch commands only
#   bash experiments/launch_scripts/launch_pca_cosine.sh               # everything
#   ONLY="pca_cos_cifar10_parity pca_cos_cifar10_parity_single" bash experiments/launch_scripts/launch_pca_cosine.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."           # plasticity/
mkdir -p logs
SCRIPT=experiments/launch_scripts/submit_stream_array.sh
COMMON=(MODE=pca PCA_COMPONENTS=6 PCA_BINS=10 PCA_SCORE=cosine OPTIMIZER=sgd LR=0.001 BATCH=256
        SEEDS="0 1 2" HEALTH_EVERY=1 EPOCHS=100)
NAMES=()

submit() {   # submit <name> <array range> <time> <model> VAR=value ...
    local name="$1" array="$2" time="$3" model="$4"; shift 4
    NAMES+=("$name")
    [[ -z "${ONLY:-}" || " $ONLY " == *" $name "* ]] || return 0
    local cmd=(env "${COMMON[@]}" MODEL_KIND="$model" RUN_TAG="${name}_${model}" "$@"
               sbatch --job-name="$name" --array="$array" --time="$time" "$SCRIPT")
    if [[ -n "${DRY_RUN:-}" ]]; then
        printf '%q ' "${cmd[@]}"; echo
    else
        echo "$name: $("${cmd[@]}")"
    fi
}
stream() { submit "$1" 0-2 24:00:00 "$2" "${@:3}" N_TASKS=600; }      # 3 seeds
single() { submit "$1" 0-17 04:00:00 "$2" "${@:3}" SINGLE=1; }         # 6 components x 3 seeds

for rule in parity upper_half; do
    stream "pca_cos_cifar10_$rule"        mlp_2x120 DATASET=cifar10 PCA_RULE="$rule"
    single "pca_cos_cifar10_${rule}_single" mlp_2x120 DATASET=cifar10 PCA_RULE="$rule"
    stream "pca_cos_mnist_$rule"          mlp_2x30  DATASET=mnist PCA_RULE="$rule"
    single "pca_cos_mnist_${rule}_single" mlp_2x30  DATASET=mnist PCA_RULE="$rule"
done

if [[ -n "${ONLY:-}" ]]; then
    for o in $ONLY; do [[ " ${NAMES[*]} " == *" $o "* ]] || echo "WARNING: ONLY names unknown experiment $o" >&2; done
fi
