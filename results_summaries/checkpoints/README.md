# Checkpoints kept in the results summary

27 files, 109 MB (billed ~2× on the `$HOME` quota). Copied by [`../scripts/collect_artifacts.sh`](../scripts/collect_artifacts.sh); the exact source path of each file is in [`../MANIFEST.tsv`](../MANIFEST.tsv). Every file is the project's standard checkpoint dict (`model_state`, `optimizer_state`, `model_kwargs`, `meta`), so from `plasticity/`:

```python
from utils.checkpoints import load_model_from_checkpoint
model = load_model_from_checkpoint("../results_summaries/checkpoints/width_sweep/mlp_2x120_seed0_task100.pt",
                                   weights_only=False)
```

These are the states the written analyses reread, or the only copies outside storage that is about to disappear. They are not the full set: complete checkpoints stay with their runs on `$HPCVAULT`, `$WORK` and `/hnvme`.

| folder | files | what they are | used by |
|---|---|---|---|
| `factor_rule/` | 3 | the pretrained mlp_3x30 encoder (seeds 0–2) that every factor-rule variant (K = 0/1/3) starts from | file 03 |
| `checkpoint_acquisition/` | 6 | Shapes3D cluster-factor stream, mlp_2x30, 100 epochs/task, seed 0: init and tasks 5, 23, 47, 83, 87 — the aged states the pilot probes | file 10 |
| `feature_acquisition/` | 6 | Result 1 sources: `init`, `partial` and `complete` encoders, synthetic mlp_2x64 (confirmation seed 100) and Shapes3D mlp_3x256 (long-decoder run, seed 0) | file 08 |
| `history_acquisition/` | 6 | Result 2, seed 0: init, the renewed ρ = 0.75 history and its persistent ρ = 0.75 control at the oldest age, for synthetic mlp_2x32 (age 40,000) and Shapes3D mlp_3x256 (age 400,000) | file 09 |
| `width_sweep/` | 3 | Shapes3D parity feature stream, mlp_2x120, seed 0: init, task 100 (before its collapse at ~146) and task 200 (after). **The rest of this sweep is only on `/hnvme`, which expires 2026-12-29.** | file 02 E2 |
| `streams/` | 3 | end of 600-task streams (task 599, seed 0): CIFAR-10 cluster mlp_2x120 (generalisation loss, no dead units), Shapes3D cluster mlp_2x30 (unit death), PCA-cosine MNIST upper half mlp_2x30 (total collapse, layer 2 dead) | file 04 |
