#!/bin/bash
# The Shapes3D feature-switch benchmark on the robot-arm datasets (data/robot3d.py): one network
# carries its weights through a stream of tasks, task n labelled by factor n % F of the dataset.
# Same budget as the Shapes3D feature stream: mlp_2x30, plain SGD (lr 0.001, batch 256), 800
# tasks x 10 epochs, seeds 0 1 2, the 480k prepared images (Shapes3D's size), 64x64x3 inputs
# (Shapes3D's D = 12 288), unit health after every task.
#
#   mpi3d_realistic_factor_stream   7 factors: object_color, object_shape, object_size, camera_height,
#                                   background_color, horizontal_axis*, vertical_axis*
#   isaac3d_factor_stream           9 factors: object_shape, robot_x*, robot_y*, camera_height, object_scale,
#                                   lighting_intensity, lighting_y_dir, object_color, wall_color
#   * the robot arm's position: a threshold in the middle (index >= K/2) instead of parity.
#
# Run launch_robot3d_single.sh first and check with visualization_scripts.single_task_solvability
# that every task is learnable alone; drop the ones that are not with TASK_FACTORS.
#
# Before the first launch: prepare the datasets as in launch_robot3d_single.sh.
# Usage (from plasticity/):
#   DRY_RUN=1 bash experiments/launch_scripts/launch_robot3d.sh           # print the sbatch commands only
#   bash experiments/launch_scripts/launch_robot3d.sh
#   ONLY=isaac3d_factor_stream TASK_FACTORS="1 2 3 4 5 6 7 8" bash experiments/launch_scripts/launch_robot3d.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."           # plasticity/
mkdir -p logs
SCRIPT=experiments/launch_scripts/submit_stream_array.sh
ONLY="${ONLY:-mpi3d_realistic_factor_stream isaac3d_factor_stream}"
COMMON=(OPTIMIZER=sgd LR=0.001 BATCH=256 SEEDS="0 1 2" HEALTH_EVERY=1 MODE=factors N_TASKS=800 EPOCHS=10)
[[ -n "${TASK_FACTORS:-}" ]] && COMMON+=(TASK_FACTORS="$TASK_FACTORS")

submit() {   # submit <name> <model> VAR=value ...
    local name="$1" model="$2"; shift 2
    [[ " $ONLY " == *" $name "* ]] || return 0
    local cmd=(env "${COMMON[@]}" MODEL_KIND="$model" RUN_TAG="${name}_${model}" "$@"
               sbatch --job-name="$name" --array=0-2 --time=24:00:00 "$SCRIPT")
    if [[ -n "${DRY_RUN:-}" ]]; then
        printf '%q ' "${cmd[@]}"; echo
    else
        echo "$name: $("${cmd[@]}")"
    fi
}

submit mpi3d_realistic_factor_stream mlp_2x30 DATASET=mpi3d_realistic
submit isaac3d_factor_stream         mlp_2x30 DATASET=isaac3d
