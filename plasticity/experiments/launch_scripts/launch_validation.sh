#!/bin/bash
# Submits the experiments that test the "shared push" explanation of loss of plasticity
# (recap doc, E1-E4) plus the controls the cluster results call for. Plain SGD (lr 0.001
# unless stated), batch 256, seeds 0 1 2, 100 epochs per task, 600 tasks, mlp_2x30 unless
# stated; every run logs input_stats and per-task unit health (run_tasks_stream --health_every 1).
#
#   E1  centre the inputs (subtract the train mean image): the shared push goes to ~0
#       e1_shapes3d_cluster_c1 (+ _single)   Shapes3D cluster factors, 50k images
#       e1_shapes3d_features_c1              the original parity stream: 432k images, 800 tasks x 10 epochs
#       e1_mnist_cluster_c1 (+ _single), e1_mnist_cluster_c1_2x120 (+ _single)
#   E2  give CIFAR-10 a Shapes3D-sized shared push (offset 0.9113: push/spread ~1,849)
#       e2_cifar10_cluster (+ _single)       the 2x30 baseline this needs
#       e2_cifar10_cluster_off1849 (+ _single)
#   E3  doses: Shapes3D centred by 0.5 and 0.9 (push/spread ~461, ~18); CIFAR-10 offsets for ~200, ~600
#   E4  shared push x lr matched to MNIST 2x30 at lr 0.001 (0.19): Shapes3D at lr 0.0001; CIFAR-10 at lr 0.01
#   M1  Shapes3D with leaky ReLU (silent units keep a gradient)
#   W1  Shapes3D at mlp_2x120 (width effect; same width as the CIFAR-10 runs)
#   G1  CIFAR-10 single tasks trained as long as the stream trains them (animal 12,000 epochs,
#       cluster factor 0 10,000 epochs, mlp_2x120): is the stream's test drop just overtraining?
#
# CIFAR-10 offsets from `uv run python -m data.input_shift --dataset cifar10 --target_ratio 200 600 1849`
# (push/spread 31.8 unshifted): 0.2772, 0.5096, 0.9113.
#
# Before the first launch (login node, from plasticity/): datasets downloaded as in launch_all.sh.
# Usage (from plasticity/):
#   DRY_RUN=1 bash experiments/launch_scripts/launch_validation.sh     # print the sbatch commands only
#   bash experiments/launch_scripts/launch_validation.sh               # everything
#   ONLY="e1_shapes3d_cluster_c1 e2_cifar10_cluster_off1849" bash experiments/launch_scripts/launch_validation.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."           # plasticity/
mkdir -p logs
SCRIPT=experiments/launch_scripts/submit_stream_array.sh
COMMON=(OPTIMIZER=sgd LR=0.001 BATCH=256 SEEDS="0 1 2" HEALTH_EVERY=1)
S3D=(DATASET=shapes3d N_SAMPLES=50000)
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
stream() { submit "$1" 0-2 24:00:00 "$2" "${@:3}" N_TASKS=600 EPOCHS=100; }      # 3 seeds
single() { submit "$1" 0-17 04:00:00 "$2" "${@:3}" SINGLE=1 EPOCHS=100; }         # 6 factors x 3 seeds

# E1: centred inputs
stream e1_shapes3d_cluster_c1        mlp_2x30  MODE=cluster "${S3D[@]}" CENTER=1
single e1_shapes3d_cluster_c1_single mlp_2x30  MODE=cluster "${S3D[@]}" CENTER=1
submit e1_shapes3d_features_c1 0-2 24:00:00 mlp_2x30 MODE=factors DATASET=shapes3d CENTER=1 N_TASKS=800 EPOCHS=10
stream e1_mnist_cluster_c1                mlp_2x30  MODE=cluster DATASET=mnist CENTER=1
single e1_mnist_cluster_c1_single         mlp_2x30  MODE=cluster DATASET=mnist CENTER=1
stream e1_mnist_cluster_c1_2x120          mlp_2x120 MODE=cluster DATASET=mnist CENTER=1
single e1_mnist_cluster_c1_2x120_single   mlp_2x120 MODE=cluster DATASET=mnist CENTER=1

# E2: a Shapes3D-sized shared push on CIFAR-10, and the 2x30 baseline it is read against
stream e2_cifar10_cluster                 mlp_2x30  MODE=cluster DATASET=cifar10
single e2_cifar10_cluster_single          mlp_2x30  MODE=cluster DATASET=cifar10
stream e2_cifar10_cluster_off1849         mlp_2x30  MODE=cluster DATASET=cifar10 OFFSET=0.9113
single e2_cifar10_cluster_off1849_single  mlp_2x30  MODE=cluster DATASET=cifar10 OFFSET=0.9113

# E3: doses of the shared push
stream e3_shapes3d_cluster_c0.5           mlp_2x30  MODE=cluster "${S3D[@]}" CENTER=0.5
stream e3_shapes3d_cluster_c0.9           mlp_2x30  MODE=cluster "${S3D[@]}" CENTER=0.9
stream e3_cifar10_cluster_off200          mlp_2x30  MODE=cluster DATASET=cifar10 OFFSET=0.2772
stream e3_cifar10_cluster_off600          mlp_2x30  MODE=cluster DATASET=cifar10 OFFSET=0.5096

# E4: shared push x lr matched to MNIST 2x30 at lr 0.001
stream e4_shapes3d_cluster_lr1e-4         mlp_2x30  MODE=cluster "${S3D[@]}" LR=0.0001
single e4_shapes3d_cluster_lr1e-4_single  mlp_2x30  MODE=cluster "${S3D[@]}" LR=0.0001
stream e4_cifar10_cluster_lr1e-2          mlp_2x30  MODE=cluster DATASET=cifar10 LR=0.01
single e4_cifar10_cluster_lr1e-2_single   mlp_2x30  MODE=cluster DATASET=cifar10 LR=0.01

# mechanism and width
stream m1_shapes3d_cluster_leaky          mlp_2x30  MODE=cluster "${S3D[@]}" ACTIVATION=leaky_relu
stream w1_shapes3d_cluster_2x120          mlp_2x120 MODE=cluster "${S3D[@]}"

# G1: CIFAR-10 single tasks trained as long as the stream trains them (one long task each)
submit g1_cifar10_animal_long 0-2 24:00:00 mlp_2x120 MODE=class DATASET=cifar10 CLASS_TASKS=animal SINGLE=1 EPOCHS=12000
submit g1_cifar10_cluster_f0_long 0-2 24:00:00 mlp_2x120 MODE=cluster DATASET=cifar10 SINGLE=1 EPOCHS=10000

if [[ -n "${ONLY:-}" ]]; then
    for o in $ONLY; do [[ " ${NAMES[*]} " == *" $o "* ]] || echo "WARNING: ONLY names unknown experiment $o" >&2; done
fi
