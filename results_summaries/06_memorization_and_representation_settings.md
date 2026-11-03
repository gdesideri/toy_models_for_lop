# 06 — Memorisation and representation plasticity settings

**Status:** code complete; **no results on the cluster** (`$HPCVAULT/plasticity_results/memorization/` and `representation/` are empty; no run directories under `plasticity/results` or `$WORK`). Some outputs exist on the Mac (e.g. the `repr_ladder_mlp_2x512_…` run the feature-visualisation notebook reads, file 01 E4). This file records what the two settings are for and the measurement issues the 2026-10-28 audit found, so that they are fixed before any run is analysed.

## 1. Research question
- **Memorisation plasticity** (`plasticity/memorization_plasticity/`): when a fraction p_change of labels is re-randomised every L steps on a fixed dataset, does the network keep memorising new labels, and does it keep predicting the old ones?
- **Representation plasticity** (`plasticity/representation_plasticity/`): when the task switches between label functions that share a controlled fraction of relevant factors (a "ladder" of feature overlap), how fast does the network adapt, and does it reuse features?

## 2. Setup
`python -m memorization_plasticity.run --dataset cifar10 --arch mlp_2x512 --p_change 0.1 --L 500`; `python -m representation_plasticity.run --stream ladder --arch mlp_2x512 --switch_every 2000`; each with its own `metrics.py`.

## 3. Results
None stored on the cluster.

## 4. References
Audit: [`research_notes/.../evidence_audit.md`](../research_notes/Plasticity%20project%20publication%20strategy/evidence_audit.md) §2 (the bullet points on the representation and memorisation runners).

## 5. Conclusions
The audit found issues that would bias results from both runners as written:
- Representation runner: evaluates on the training data, logs the absolute global step, merges all recurrences of a task under one name, and its `adaptation_speed` returns the raw step instead of steps since the switch; the AUC integrates across discontinuous occurrences; training slices logits to the current class count but evaluation uses all logits.
- The ladder declares *pairs* of relevant factors and an overlap of 1/3, but the label function uses only one factor of each pair, so the declared overlap is not the overlap of predictive labels. EWC's Fisher is estimated on the *new* task's loader right after a switch.
- Memorisation runner: `previous_label_prediction_rate` measures the *original* labels, not the immediately previous ones; the changed/unchanged masks mean "currently different from original", not "ever changed".

## 6. Next steps proposed
- *(From the audit)* Fix the metric definitions above before running; treat the memorisation-vs-representation taxonomy as a supporting control rather than a separate main story.
