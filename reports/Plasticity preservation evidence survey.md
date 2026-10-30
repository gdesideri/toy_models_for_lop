# What plasticity preserving methods actually preserve

Evidence survey checked on 29 October 2026. This is a focused review of major method families and selected 2025–2026 extensions, not a common-protocol leaderboard or an exhaustive systematic review.

## Main conclusion

Several methods have substantial evidence for sustained arbitrary-label fitting and structured prediction. It is inaccurate to characterize the field as preserving memorization alone. Published limitations are often specific to architecture, optimizer, regularization, or evaluation protocol. Absence of a matched new-feature or ordering experiment is an evidence gap, not a demonstrated failure.

The proposed research questions—task structure, feature demand, input versus rule changes, and history—are not invariance requirements that every method must satisfy. Useful transfer can legitimately depend on order. The relevant failure is impaired future acquisition under an appropriate fresh baseline, not merely different weights or forgotten old tasks.

## Primary method evidence

### Continual backpropagation

The original study demonstrates sustained supervised and reinforcement learning, including Continual ImageNet, incremental CIFAR-100 and permuted MNIST. It also reports beneficial transfer. These results argue against a blanket claim that neuron replacement necessarily prevents feature reuse. Its current utility does not solve forgetting; forgetting is a separate outcome from acquisition impairment.

Source: [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7), especially Maintaining plasticity, Discussion, and Methods.

### ReDo and self normalized resets

ReDo recycles dormant units. SNR estimates neuron-specific inactivity relative to nominal firing behavior. The SNR study compares these with CBP and regularizers across memorization and structured tasks. Its transformer results identify a concrete scope limit: feedforward-only resets do not directly address attention pathology. The paper's transformer tables and tuning appendix should be consulted before generalizing its rankings.

Source: [Self-Normalized Resets](https://arxiv.org/html/2410.20098v3), Sections 4.1–4.2, Tables 1–3 and Appendix C; [ReDo](https://proceedings.mlr.press/v202/sokar23a.html).

### Deep Fourier features

DFF is supported by both trainability and structured generalization experiments. The final paper considers label noise, class growth, permutations and larger image datasets. Its regularization sensitivity analysis distinguishes fitting ability from test performance. Unregularized overfitting is not evidence that the properly regularized method loses trainability.

Source: [Plastic Learning with Deep Fourier Features, ICLR 2025 final paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/4d42358702dff82e1436550a05ade260-Paper-Conference.pdf), Sections 5.1–5.2 and Appendix D.

### Normalization and effective learning rate control

NaP combines preactivation normalization and weight-norm projection. It identifies effective-learning-rate changes as an important qualification to normalization-based plasticity. Its reinforcement-learning results also show the importance of an explicit learning-rate schedule; fixed effective learning rate is not automatically the best end-to-end training policy.

Source: [Normalization and effective learning rates in reinforcement learning](https://arxiv.org/html/2407.01800), Sections 3–5 and sequential-ALE experiment details.

### Shrink and perturb and initialization regularization

These remain important baselines, not universally reliable solutions. Regularization can protect optimization while constraining current-task fitting. A modern activation comparison reports a particularly adverse interaction between DFF and L2-Init on alternating hard/easy CIFAR tasks. That is a protocol-specific combination failure, not a universal DFF failure or proof that all initialization regularization is harmful.

Sources: [Activation Function Design Sustains Plasticity](https://arxiv.org/html/2509.22562), Appendix E, especially E7–E9; [Learning Continually by Spectral Regularization](https://arxiv.org/html/2406.06811), Sections 5.1–5.3 and Appendix C.

### Spectral and isometry based methods

Spectral regularization directly targets spectral scale rather than anchoring every parameter. Its study covers multiple nonstationarities and architectures. L2-ER combines effective-rank and L2 regularization in a newer curvature account. Parseval regularization has primarily continual-RL evidence. AdamO adds a newer isometry-promoting adaptive optimization approach with both arbitrary-label and held-out structured evaluations. None of these results alone establishes a guarantee for all semantic feature requirements or histories.

Sources: [Spectral regularization](https://arxiv.org/html/2406.06811); [Spectral Collapse, ICML 2026](https://proceedings.mlr.press/v306/prakash26b.html); [Parseval regularization](https://arxiv.org/html/2412.07224); [Dynamical isometry and AdamO](https://arxiv.org/html/2606.09762).

### PLASTIC

PLASTIC already makes the input/label distinction central. Its synthetic tasks are a growing CIFAR buffer and class-consistent label remapping. These are relevant precedents, but do not isolate acquisition of a previously unused semantic factor. Its combined design supports the view that different adaptation demands can need different ingredients.

Source: [PLASTIC, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/c464fc4516aca4e68f2a14e67c6f0402-Paper-Conference.pdf), Figure 2 and Section 3.

### Smooth leaky activation designs

The newer activation study includes extensive tuning, supervised streams and changing MuJoCo settings. It reports good results but explicitly distinguishes trainability, generalization and numerical stability. Its class-incremental accuracy uses replay and mixes acquisition with retention; it cannot be treated as a pure fresh-relative plasticity measure.

Source: [Activation Function Design Sustains Plasticity](https://arxiv.org/html/2509.22562), Sections 6–7 and Appendices B, E and F. A final ICLR 2026 paper is available through [OpenReview](https://openreview.net/forum?id=XZf6wObHX4).

## What is established about the proposed properties

1. **Memorization versus reusable structure:** Both have been tested by strong methods. The unresolved comparison is controlled future learning after histories with independently varied memorization demand, not the existence of structured benchmarks.
2. **Feature reuse versus new feature demand:** New-class and new-image results provide meaningful evidence of acquisition, but do not by themselves implement matched feature-demand experiments. Feature introduction should be verified through frozen-representation probes and fresh difficulty controls.
3. **Input versus labeling-rule changes:** This difference already has direct evidence and precedent. The scope of the two protocols must be stated: growing data, pixel reencoding, class remapping and instance-random labels are distinct constructions.
4. **History beyond exposure and recency:** Random task orders and multi-seed evaluations do not substitute for a matched-prefix/common-suffix comparison. This review did not find sufficient direct evidence to label the major methods successful or unsuccessful under that precise test.
5. **Prospective prediction across demands:** This is a separate claim from either a method's good performance or disagreement among benchmark rankings. The proposed diagnostic panel must be frozen and tested on genuinely held-out task families, with simple learning-curve baselines and existing diagnostics.

## Consequence for the project

The evidence supports the selected claim that one conventional test need not characterize all future learning. It only partly supports the claim that this project's chosen stream properties explain the differences: architecture and optimization can also explain published failures. The claim that a compact diagnostic predicts untouched future demands remains a research hypothesis.

A strong project would identify a controlled mismatch, explain its source, and prospectively predict its presence or absence. Successful preservation methods are essential positive controls. The contribution should survive even when several methods eliminate the barrier.

Predictive work must also compare against [Optimization Readiness](https://arxiv.org/html/2605.09044v1), rather than treating prediction from checkpoint measurements as an unexplored idea.

## Limitations of this survey

Benchmarks, tuning budgets, architecture sizes, losses and observation protocols differ. Reported weak performance does not imply a universal failure, and superiority on a training metric does not establish better generalization. Some recent comparisons require independent reproduction. The review's parallel research stopped because the workspace reached its credit limit; key claims included here were subsequently checked directly against primary sources. No experiment was run for this survey.
