# Experiment protocol (v0.1)

## Core questions

1. How accurate is the critic's prediction of the benefit of an actor update?
2. Which critic errors matter: absolute Q-value error, incorrect action rankings, or error projected along the policy update?
3. How do sampled and full-action gradients differ in update quality?
4. Does backtracking to cap per-state KL reduce harmful steps, or simply make every update smaller?

## Measurement definitions

Fix an old policy \(\pi\), the learned critic \(\hat Q\), and a visited state \(s\). Construct candidate \(\pi'\) changing **only that state**.

- **Predicted local gain:** \(\hat\Delta_s=\sum_a(\pi'(a|s)-\pi(a|s))\hat Q^\pi(s,a)\).
- **True local gain:** \(\Delta_s=\sum_a(\pi'(a|s)-\pi(a|s))Q^\pi(s,a)\).
- **Actual global gain:** \(\Delta J=J(\pi')-J(\pi)\), evaluated exactly from the start distribution.
- **Realized state KL:** \(D_{KL}(\pi'(\cdot|s)\|\pi(\cdot|s))\).
- **Critic RMSE:** uniform average over all decision-state/action pairs. It is *not* training-distribution-weighted.
- **Ranking accuracy:** fraction of non-tied true action pairs with correctly signed critic Q difference.
- **Regret:** uniform over decision states of \(\max_a Q^\pi(s,a)-Q^\pi(s,\arg\max_b \hat Q(s,b))\).

For the tabular actor, each individual update changes only one state. By the performance-difference lemma, positive true local gain at that state implies nonnegative global improvement, provided it has nonzero discounted occupancy under the new policy. **Thus, for these single-state counterfactuals, opposite signs of true local gain and global gain would indicate a bug.** An earlier general intuition that local improvement can coincide with global degradation does not apply to this restricted case. Later neural policies may change actions at many states simultaneously, which requires a different analysis.

## Counterfactual comparison

At each QAC checkpoint, compute six *non-committed* candidate actor updates from identical logits and critic parameters:

- Three operators: sampled QAC, full-action QAC, KL-backtracked full-action QAC.
- Two Q sources: learned critic and exact/oracle Q for analysis only.

The actual training trajectory follows the configured method using learned estimates only. The candidate updates are computed solely for measurement and never alter the learner. The oracle is therefore a diagnostic upper-control, not a deployable method.

## First-analysis checks

- Plot predicted local gain versus true local gain, marking critic-ranking error.
- Plot actual gain versus realized KL, stratified by critic quality.
- Compute the fraction of candidate updates with negative global gain (learned versus oracle).
- Compare paired candidate updates at matched checkpoints; do not treat these pairs as independent samples.
- Compare full training curves using multiple independent seeds, reporting dispersion and sample efficiency.
- Inspect candidate moves with near-zero Q-value gaps; rankings may be practically irrelevant even if formally incorrect.

## Confounding and limitations

- The three QAC training runs start with equivalent configuration but generate **different trajectories once policies diverge**. Only the recorded candidate updates are strictly same-checkpoint comparisons.
- The fixed actor step size and KL cap do not imply matched realized update sizes. To attribute benefits to a *constraint* rather than simply a smaller learning rate, sweep actor learning rate and compare at matched realized KL distributions.
- Tabular critics do not have function-approximation generalization error; this is a mathematical/debugging baseline for the neural phase.
- One-step Expected SARSA learns from observed actions. Unvisited actions can remain inaccurate, especially early in training.
- Actual policy performance here is exact *expected discounted return*, not episodic average return. Later benchmarks should additionally report empirical evaluation-return distributions.
- There is one handcrafted environment. Its results are not evidence of generality.

## Next experimental improvement

Introduce controlled critic perturbations at frozen checkpoints while preserving exact reference Q. Compare random, action-dependent, and structured biases. Only after characterizing those should neural critics and more sophisticated actor-improvement operators be added.
