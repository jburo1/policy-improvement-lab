# Experiment protocol

## First experiment

[Experiment 001](../experiments/001-critic-reliability/README.md) asks how reliably one full-action QAC learner's critic predicts the benefit of a hypothetical policy update. The committed trajectory uses learned estimates only. The primary comparison is full-action updates with learned versus exact critic guidance at the same frozen checkpoints.

Sampled-action comparisons, KL studies, actor-rate sweeps, and controlled critic corruption are later work. Existing implementations remain available; they are not additional experimental conditions for 001.

## Learning and checkpoint semantics

At a transition, freeze the current logits and critic estimates. Sample an action from that policy. Expected SARSA uses

\[
y=r+\gamma\sum_b\pi(b|s')\hat Q(s',b),\qquad
\hat Q(s,a)\leftarrow\hat Q(s,a)+\beta(y-\hat Q(s,a)),
\]

with no bootstrap on termination. Both the actor direction and this target read the **pre-transition-update critic**. Full-action QAC changes the visited logit row by

\[
\theta_{s,a}\leftarrow\theta_{s,a}+\alpha\pi(a|s)
\left(\hat Q(s,a)-\sum_b\pi(b|s)\hat Q(s,b)\right).
\]

This is the gradient of the local frozen-critic objective \(\sum_a\pi(a|s)\hat Q(s,a)\). The sampled score-function gradient with a state baseline has the same action expectation. This does not make finite sampled and full-action updates equivalent, nor imply that their complete training trajectories coincide. These online local updates do not include explicit discounted state-visitation weighting.

One-step actor–critic similarly uses the pre-update TD error \(r+\gamma\hat V(s')-\hat V(s)\), omitting the bootstrap on termination.

Checkpoint `step=k` means **after k completed environment transitions and their actor/critic updates**. Record step 0, every evaluation interval, and the final step even when off interval. All checkpoint return, critic-quality, and counterfactual measurements use that same policy and critic. The diagnostic state is the next decision state (reset from the initial distribution after termination). A separate seeded diagnostic RNG samples an action for sampled-action candidates; it does not consume training randomness. The checkpoint state is a visited state, not a uniform draw over states.

## Measurements

Fix policy \(\pi\), learned critic \(\hat Q\), and checkpoint state \(s\). A candidate \(\pi'\) changes only that state's logits.

- **Predicted local gain:** \(\hat\Delta_s=\sum_a(\pi'(a|s)-\pi(a|s))Q_{\mathrm{guidance}}(s,a)\), where guidance is learned or exact as labeled.
- **True local gain:** \(\Delta_s=\sum_a(\pi'(a|s)-\pi(a|s))Q^\pi(s,a)\).
- **Actual global gain:** \(\Delta J=J(\pi')-J(\pi)\), evaluated exactly from the fixed initial distribution.
- **Realized state KL:** \(D_{KL}(\pi'(\cdot|s)\|\pi(\cdot|s))\).
- **Critic RMSE:** uniform over decision-state/action pairs, not visitation weighted.
- **Ranking accuracy:** fraction of true action pairs with gap greater than `1e-6` whose critic differences have the correct sign. A critic tie is incorrect; no eligible pairs yields NaN.
- **Selection regret:** uniform over decision states of \(\max_a Q^\pi(s,a)-Q^\pi(s,\arg\max_b\hat Q(s,b))\); `argmax` breaks ties by the first action.

Critic-quality columns always describe the **learned critic**, including on oracle candidate rows. `predicted_local_gain` on oracle rows uses exact guidance, and therefore equals `true_local_gain` up to numerical error.

For a single-state change, the performance-difference identity is

\[
J(\pi')-J(\pi)=d_{\pi'}(s)\Delta_s,\qquad
d_{\pi'}^\top=\mu^\top(I-\gamma P_{\pi'})^{-1}.
\]

Here \(d\) is unnormalized discounted occupancy under the **new** policy. Opposite nonzero signs of local and global gain indicate a bug, allowing for floating-point tolerance; an unreachable state has zero global gain. This restricted identity does not directly apply to neural updates changing many states at once.

## Counterfactual provenance and oracle separation

The existing exporter emits six uncommitted candidates at each QAC checkpoint: three operators times two critic sources. Experiment 001 analyzes only rows with `training_method=qac_full` and candidate `method=qac_full`; the other candidates are outside its scope.

Within a run directory, join candidates to checkpoints on `(training_method, seed, step)` versus `(method, seed, step)`. `method` on a candidate is the hypothetical operator, **not** its originating learner. `checkpoint_expected_return` repeats the originating policy's exact return for an explicit consistency check. Pair learned/oracle rows on `(training_method, seed, step, state, action, method)`. Keep run directories/configurations distinct when combining results.

Earlier exports lack originating-method provenance and used different checkpoint timing; regenerate them before using this protocol or the paired plot.

All candidate operators receive the same frozen policy and their labeled critic guidance. Exact values evaluate the old policy and candidate performance only. Candidates return new arrays and never alter training logits, critics, or random state. The exact critic is a reference control, not a guaranteed best finite-step update or a deployable learning method.

## Analysis and limitations

Predeclare settings and retain every seed. Summarize gain prediction error, negative-global-gain frequency (using a numerical zero tolerance), and paired full-action learned/oracle differences. Treat repeated checkpoints and candidate pairs as dependent; compute seed-level summaries before reporting across-seed dispersion. Plot training return as context, not as a comparison between two learners.

The shared plot command produces training mean ± sample SD (zero width for one seed), learned-guided gain scatter plots across available candidate operators, and a paired full-action learned/oracle global-gain plot. The latter is the primary paired visualization for 001; the other operator scatters are supplementary plumbing outputs, not broader experimental comparisons. CSVs retain all measurements for the planned seed-level analysis.

Only one handcrafted MDP and a tabular critic are used. Unvisited actions can remain inaccurate; no neural generalization error is modeled. Fixed nominal actor rates do not match realized KL across critic sources. Checkpoint states change along the trajectory, and quality summaries are uniformly weighted while candidate states are visited. Exact expected discounted return is not empirical episodic return. No broad algorithm or conservative-update claims follow from this experiment.
