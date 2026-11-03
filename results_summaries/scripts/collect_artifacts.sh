#!/bin/bash
# Copy the figures, summary CSVs and the few checkpoints the results summaries
# cite into results_summaries/{figures,csv,checkpoints}, and write MANIFEST.tsv
# (destination, source, bytes) so every copied file can be traced back.
#
#     bash results_summaries/scripts/collect_artifacts.sh
#
# Reads from the vault, $WORK and /hnvme, so run it on a login node (GPU nodes
# mount none of the first two). The /hnvme workspace expires 2026-12-29: the
# width-sweep checkpoints copied here are the only ones outside it.
set -euo pipefail
cd "$(dirname "$0")/../.."
V=${HPCVAULT:-/home/vault/v170be/v170be10}/plasticity_results
W=${WORK:-/home/atuin/v170be/v170be10}/plasticity_results
H=/hnvme/workspace/v170be10-plasticity_results
L=plasticity/results
S=results_summaries
MANIFEST=$S/MANIFEST.tsv
printf 'destination\tsource\tbytes\n' > "$MANIFEST"

put() {  # put <source> <destination relative to results_summaries/>
    local src=$1 dst=$S/$2
    if [ ! -e "$src" ]; then echo "MISSING $src" >&2; return; fi
    mkdir -p "$(dirname "$dst")"
    cp -p "$src" "$dst"
    printf '%s\t%s\t%s\n' "$2" "$(readlink -f "$src")" "$(stat -L -c %s "$src")" >> "$MANIFEST"
}
latest() { ls -d $1 2>/dev/null | sort -V | tail -1; }  # last match of a glob, version-sorted

FA=$V/feature_acquisition
FA_SYN_CONF=$FA/synthetic_mlp_2x64_synthetic_confirmation-1f5b89ef16be
FA_SYN_DISC=$FA/synthetic_mlp_2x64_synthetic_discovery-81f6421930f1
FA_S3D_LONG=$FA/shapes3d_mlp_3x256_shapes3d_mlp_longdec-1c8038f4e44c
HA=$V/history_acquisition
HA_W32=$HA/synthetic_mlp_2x32_width32-726e7934832f
HA_S3D=$HA/shapes3d_mlp_3x256_discovery-6a583ae81d27
CA=$V/checkpoint_acquisition/shapes3d_cluster_stream_ep100_subset__stream_shapes3d_mlp_2x30_perm000_clu6x8_d50_seed0__pilot-71770d79cfe9

# --- figures: 02 Shapes3D feature / permuted streams --------------------------
put $V/stream_test_accuracy_width_ma.png                        figures/02_width_sweep_test_accuracy.png
put $V/stream_alive_units_width_ma_n20000_layer1.png            figures/02_width_sweep_alive_units_layer1.png
put $V/stream_killed_units_width_n20000_layer1.png              figures/02_width_sweep_killed_units_layer1.png
put $V/shapes3d_perm100/azimuth_width_sweep.png                 figures/02_azimuth_width_sweep.png
put $V/shapes3d_reprdist_features_sgd/feature_pair_distance.png figures/02_feature_pair_distance.png
# --- figures: 03 factor-rule stream -------------------------------------------
put $V/factor_rule_pretrain_factors.png                         figures/03_factor_rule_pretrain_factors.png
put $V/factor_rule_probe_factors.png                            figures/03_factor_rule_probe_factors.png
put $V/factor_rule_stream_modes_500.png                         figures/03_factor_rule_modes_500.png
put $V/factor_rule_stream_modes_single_500.png                  figures/03_factor_rule_modes_single_500.png
put $V/factor_rule_stream_modes_single_10ep.png                 figures/03_factor_rule_modes_single_10ep.png
put $V/factor_rule_stream_alive_units_500.png                   figures/03_factor_rule_alive_units_500.png
put $V/factor_rule_stream_alive_units_10ep.png                  figures/03_factor_rule_alive_units_10ep.png
# --- figures: 07 MPI3D --------------------------------------------------------
put $V/mpi3d_realistic_test_accuracy_width_ma.png               figures/07_mpi3d_test_accuracy_width.png
put $V/mpi3d_realistic_alive_units_width_ma_layer1.png          figures/07_mpi3d_alive_units_layer1.png
# --- figures: 08 feature acquisition (Result 1) -------------------------------
put $FA_SYN_CONF/figures/full_minus_frozen.png                  figures/08_synthetic_confirmation_full_minus_frozen.png
put $FA_SYN_CONF/figures/acquisition_curves.png                 figures/08_synthetic_confirmation_acquisition_curves.png
put $FA_SYN_CONF/figures/final_accuracy.png                     figures/08_synthetic_confirmation_final_accuracy.png
put $FA_S3D_LONG/figures/full_minus_frozen.png                  figures/08_shapes3d_longdec_full_minus_frozen.png
put $FA_S3D_LONG/figures/final_accuracy.png                     figures/08_shapes3d_longdec_final_accuracy.png
# --- figures: 09 history acquisition (Result 2) -------------------------------
for f in deficit_vs_corruption deficit_vs_rule deficit_vs_block_length frozen_vs_full; do
  put $HA_W32/analysis/main-c31a19c7ae2b/figures/$f.png         figures/09_synthetic_w32_$f.png
done
for f in deficit_vs_corruption deficit_vs_block_length; do
  put $HA_S3D/analysis/main-60d2d72208b7/figures/$f.png         figures/09_shapes3d_3x256_$f.png
done
# --- figures: 10 checkpoint acquisition pilot ---------------------------------
for f in aged_minus_fresh frozen_vs_full curves_extended late_minus_early; do
  put $CA/figures/$f.png                                        figures/10_pilot_$f.png
done

# --- summary CSVs / JSON ------------------------------------------------------
# (the per-branch summary.csv / acquisition_stats.csv files are 1-5 MB of
# per-seed curves; make_figures.py derives compact tables from them instead)
put $V/shapes3d_reprdist_features_sgd/feature_pair_distance.json csv/shapes3d_feature_pair_distance.json
put $V/shapes3d_perm100/azimuth_width_sweep.json                csv/shapes3d_azimuth_width_sweep.json
put $V/shapes3d_factor_rule_stream/factorrule_shapes3d_mlp_3x30_perm000_seed0/summary.json csv/factor_rule_seed0_summary.json
for run in $FA/*/; do
  tag=$(basename "$run"); tag=${tag%-*}
  put "$run/comparisons.csv"  "csv/feature_acquisition/${tag}_comparisons.csv"
done
for run in $HA/*/; do
  tag=$(basename "$run"); tag=${tag%-*}
  a=$(ls -d "$run"/analysis/main-* | head -1)
  put "$a/contrasts.csv"          "csv/history_acquisition/${tag}_contrasts.csv"
  put "$a/report.md"              "csv/history_acquisition/${tag}_report.md"
done
put $CA/summary.csv       csv/checkpoint_acquisition_summary.csv
put $CA/comparisons.csv   csv/checkpoint_acquisition_comparisons.csv

# --- crucial checkpoints ------------------------------------------------------
# factor-rule stream: the pretrained, frozen encoder every K = 0/1/3 run starts from
for s in 0 1 2; do
  put $L/shapes3d_factor_rule_stream/factorrule_shapes3d_mlp_3x30_perm000_seed$s/checkpoints/encoder.pt \
      checkpoints/factor_rule/encoder_mlp_3x30_seed$s.pt
done
# checkpoint-acquisition pilot: the aged Shapes3D cluster-stream states it probes
P=$L/shapes3d_cluster_stream_ep100_subset/stream_shapes3d_mlp_2x30_perm000_clu6x8_d50_seed0
put $P/checkpoints/init.pt checkpoints/checkpoint_acquisition/shapes3d_cluster_mlp_2x30_seed0_init.pt
for t in 005 023 047 083 087; do
  put $P/task_$t/checkpoints/epoch_100.pt checkpoints/checkpoint_acquisition/shapes3d_cluster_mlp_2x30_seed0_task$t.pt
done
# Result 1 sources: init, partial (supervised on some factors) and complete encoders
for v in init partial complete; do
  put "$(latest "$FA_SYN_CONF/source/seed_100/${v}*.pt")" checkpoints/feature_acquisition/synthetic_mlp_2x64_seed100_${v}.pt
  put "$(latest "$FA_S3D_LONG/source/seed_0/${v}*.pt")"   checkpoints/feature_acquisition/shapes3d_mlp_3x256_seed0_${v}.pt
done
# Result 2: the largest-deficit history (renewed, rho 0.75) next to its persistent control
for h in init.pt f1_renewed_rho0.75_L2000/checkpoints/age_40000.pt f1_persistent_rho0.75/checkpoints/age_40000.pt; do
  name=$(echo "$h" | sed 's|/checkpoints/|_|; s|\.pt$||')
  put "$HA_W32/histories/seed_0/$h" "checkpoints/history_acquisition/synthetic_w32_seed0_${name}.pt"
done
put "$HA_S3D/histories/seed_0/init.pt" checkpoints/history_acquisition/shapes3d_3x256_seed0_init.pt
for c in renewed_rho0.75 persistent_rho0.75; do
  src=$(latest "$HA_S3D/histories/seed_0/*${c}*/checkpoints/age_*.pt")
  put "$src" "checkpoints/history_acquisition/shapes3d_3x256_seed0_$(basename "$(dirname "$(dirname "$src")")")_$(basename "$src")"
done
# Shapes3D width sweep (only on the expiring /hnvme workspace): mlp_2x120 seed 0
# before (task 100) and after (task 200) its collapse at task ~146
WS=$H/shapes3d_stream_features_width_sgd/stream_shapes3d_mlp_2x120_perm000_features012345_parity_seed0
put $WS/checkpoints/init.pt              checkpoints/width_sweep/mlp_2x120_seed0_init.pt
put $WS/task_100/checkpoints/epoch_010.pt checkpoints/width_sweep/mlp_2x120_seed0_task100.pt
put $WS/task_200/checkpoints/epoch_010.pt checkpoints/width_sweep/mlp_2x120_seed0_task200.pt
# class / cluster / PCA streams at the end of 600 tasks
put $W/cifar10_cluster_stream_mlp_2x120/stream_cifar10_mlp_2x120_perm000_clu6x8_d50_seed0/task_599/checkpoints/epoch_100.pt \
    checkpoints/streams/cifar10_cluster_mlp_2x120_seed0_task599.pt
put $W/shapes3d_cluster_stream_mlp_2x30/stream_shapes3d_mlp_2x30_perm000_clu6x8_d50_seed0/task_599/checkpoints/epoch_100.pt \
    checkpoints/streams/shapes3d_cluster_mlp_2x30_seed0_task599.pt
put $W/pca_cos_mnist_upper_half_mlp_2x30/stream_mnist_mlp_2x30_perm000_pca6_bins10_cos_upper_half_seed0/task_599/checkpoints/epoch_100.pt \
    checkpoints/streams/pca_cos_mnist_upper_half_mlp_2x30_seed0_task599.pt

echo "wrote $(($(wc -l < "$MANIFEST")-1)) files; manifest: $MANIFEST"
