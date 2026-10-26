# 001 — Critic Reliability


## Motivation

Actor–critic reinforcement learning uses learned value estimates to guide policy updates. These estimates are noisy,particularly in the early stages of training when experience is limited.

Accurate value estimation and reliable policy improvement are not necessarily equivalent. A critic may exhibit substantial value-prediction error while preserving the relative action preferences needed to improve a policy. Conversely, errors in those preferences may cause the actor to reinforce inferior actions, potentially reducing performance.

This experiment investigates the relationship between **critic estimation quality** and the **reliability of critic-guided policy updates**. By reliability, we mean whether the policy update actually improves the policy.

To this end, we construct a small, exactly evaluable Markov decision process that provides access to the true action values and expected return of any policy. This makes it possible to distinguish errors in the learned critic from their consequences for policy improvement. This controlled setting isolates the relationship between critic error and policy improvement before introducing additional complexity from function approximation and larger environments.

## Research Question

**How reliably do learned action-value estimates guide beneficial policy improvements, relative to exact action values, in a simple tabular QAC RL training run?**

The investigation focuses on three aspects:

1. **Prediction accuracy:** How accurately does the learned critic predict the improvement produced by its own policy update?
2. **Update reliability:** How frequently do critic-guided updates increase or decrease true policy performance?
3. **Critic quality:** How do estimation error and action-ranking accuracy relate to the quality of policy improvements?

## Methodology

### 1. Experimental Environment

The experiment uses `branching_mdp_v1`, a handcrafted finite discounted MDP with stochastic transitions and delayed rewards.

| Property | Configuration |
|---|---|
| Environment | `branching_mdp_v1` |
| Decision states | 5 |
| Actions per state | 3 |
| Terminal states | 1 absorbing state |
| Discount factor | 0.95 |
| Initial state | State 0 |
| Reward structure | Transition costs and delayed terminal reward |
| Reference evaluation | Exact Bellman solution |

The finite state space permits exact computation of $Q^\pi(s,a)$ and expected discounted return $J(\pi)$ for every checkpoint policy.

### 2. Learning Algorithm

Training uses a tabular **full-action Q Actor–Critic (QAC)** learner.

The critic estimates action values using on-policy one-step Expected SARSA. The actor performs gradient ascent on the expected action value under the current policy:

$$
\theta_{s,a} \leftarrow \theta_{s,a} +
\alpha \pi(a|s)
\left[
\hat Q(s,a) -
\sum_b \pi(b|s)\hat Q(s,b)
\right].
$$

| Parameter | Configuration |
|---|---|
| Training method | `qac_full` |
| Policy representation | Tabular softmax logits |
| Critic representation | Tabular action values |
| Critic update | One-step Expected SARSA |
| Actor learning rate | 0.20 |
| Critic learning rate | 0.12 |
| Initialization | Zero actor logits and critic values |
| Training duration | 5,000 environment transitions per seed |
| Checkpoint interval | 100 transitions |
| Random seeds | 0–9 |


### 3. Counterfactual Policy Updates

At each checkpoint, the current policy and learned critic are frozen.

Two hypothetical full-action QAC updates are constructed at the checkpoint's next decision state.

| Condition | Learned guidance | Oracle guidance |
|---|---|---|
| Starting policy | Current checkpoint policy | Same |
| Update operator | Full-action QAC | Full-action QAC |
| Critic guidance | **Learned $\hat Q$** | **Exact $Q^\pi$** |
| Actor learning rate | 0.20 | 0.20 |
| Updated states | One | Same |

Both candidates are evaluated against the same starting policy. This paired comparison isolates the difference between learned and exact critic guidance at a fixed checkpoint.

### 4. Measurements

Three categories of measurements characterize critic quality and policy-improvement reliability.

**Critic quality**

| Metric | Description |
|---|---|
| RMSE | Error between learned and exact action values |
| Action-ranking accuracy | Fraction of eligible action pairs correctly ordered by the learned critic |
| Action-selection regret | True value lost by selecting the learned critic's preferred action rather than the optimal action |
| Gradient alignment | Cosine between learned-critic and exact-critic full-action logit gradients at the evaluated state, plus both norms |

**Candidate update quality**

| Metric | Description |
|---|---|
| Predicted local gain | Estimated improvement under the guidance critic |
| True local gain | Local improvement evaluated using exact action values |
| Actual performance gain | Change in exact expected return, $J(\pi')-J(\pi)$ |
| Realized state KL | $D_{\mathrm{KL}}(\pi'(\cdot\mid s)\Vert\pi(\cdot\mid s))$ |
| Signed / absolute prediction error | Predicted minus true local gain, and its absolute value, for learned-guided updates |
| Update outcome | Beneficial, harmful, or effectively zero global gain using absolute tolerance $10^{-12}$ |

**Training performance**

| Metric | Description |
|---|---|
| Expected return | Exact discounted return of the trained policy at each checkpoint |
| Training progress | Number of completed environment transitions |
| Optimal expected return | Exact optimal MDP return $J^*$ as a convergence reference |

All measurements follow the definitions in the [shared experimental protocol](../../docs/experiment_protocol.md).

### 5. Analysis

The primary analysis compares learned-guided and oracle-guided candidate updates at matched checkpoints.

The plotting command includes only `qac_full` training checkpoints and candidates with both `training_method=qac_full` and `method=qac_full`. Other diagnostic operators remain in the raw CSVs but are excluded from every analysis output. Learned and oracle candidates are paired on originating training method, seed, step, state, action, and candidate operator; missing or duplicate pairs are errors. Oracle guidance is a diagnostic control, not a second training trajectory.

| Analysis | Purpose |
|---|---|
| Predicted versus true local gain | Characterize the accuracy of the learned critic's improvement predictions |
| Update outcome counts and frequencies | Quantify beneficial, harmful, and effectively zero learned-guided updates |
| Paired learned/oracle gain and KL | Compare improvement and magnitude at checkpoints and within each seed |
| Critic quality versus update error | Examine relationships between critic estimation quality and improvement reliability |
| Critic quality and gradient alignment over training | Track value error, action preferences, and local gradient direction |
| Signed and absolute prediction errors over training | Distinguish overestimation from underestimation and error magnitude |
| Expected return over training | Contextualize convergence relative to $J^*$ |

Measurements are summarized within each seed before computing across-seed means and sample standard deviations. Checkpoints are repeated observations from the same training trajectory and are not treated as independent experimental replicates.

The summary uses each seed's last checkpoint for final expected return. Other summary statistics average all recorded checkpoints within each seed, including step 0 and the final checkpoint. Signed prediction error is `predicted_local_gain - true_local_gain` on learned-guided candidates: positive is overestimation and negative is underestimation. Absolute error measures magnitude without cancellation. Beneficial updates have global gain above `1e-12`, harmful updates below `-1e-12`, and effectively zero updates have absolute gain at most `1e-12`. Counts and frequencies use all recorded learned-guided full-action candidates, including the initial zero update. This numerical tolerance does not imply statistical significance. Learned/oracle gain and state-KL summaries use matched candidates; paired differences are oracle minus learned.

Gradient alignment compares the full-action local logit gradients before either hypothetical update, with no learning-rate or state-occupancy factor. Cosine is undefined (NaN) if either Euclidean norm is at most `1e-12`. Both norms are saved; undefined cosines are excluded from averages, with defined counts/fractions and zero-gradient counts reported. An entirely undefined checkpoint appears as a gap in the cosine curve. No defined observations in a seed leaves its mean cosine blank. Oracle measurements remain diagnostic-only.

The outcome-frequency plot first computes all three frequencies for each seed in fixed training-step bins `[0, 500]`, `(500, 1000]`, and so on, then plots the mean ± sample SD across seeds at each bin's upper edge. Seeds with no observations in a bin do not contribute to that bin; empty bins are omitted. Training, critic-quality, gradient-alignment, and prediction-error curves similarly show mean ± sample SD across seeds at each step. Single-seed plot bands have zero width; single-seed summary SDs are undefined and left blank in the CSV. Scatter plots show checkpoint observations descriptively, without treating them as independent replicates or estimating checkpoint-based confidence intervals. Labeled seed means in the paired gain and KL figures expose between-seed differences.

The training reference $J^*$ uses offline policy iteration with exact Bellman policy evaluation and an optimality-residual tolerance of `1e-12`. It uses the same discount and initial distribution as the training return. It is never passed to learning.

Associations between critic quality and update performance are descriptive rather than causal.

## Reproduction

Run from the repository root. Install the analysis dependencies once:

```bash
uv sync --locked --python 3.11 --extra analysis
```

The following commands reproduce the experiment's raw measurements and complete offline analysis.

### 1. Run the experiment

```bash
uv run --locked policy-lab \
  --methods qac_full \
  --steps 5000 \
  --eval-every 100 \
  --actor-lr 0.20 \
  --critic-lr 0.12 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --out-dir experiments/001-critic-reliability/data
git rev-parse HEAD > experiments/001-critic-reliability/data/source_revision.txt
```
### 2. Generate figures and statistical summaries

```bash
uv run --locked policy-lab-plot \
  experiments/001-critic-reliability/data \
  --out-dir experiments/001-critic-reliability/analysis
```

Plotting reads the previously collected CSVs and never reruns training or modifies raw measurements. Regenerate older runs without `learned_gradient_norm`, `oracle_gradient_norm`, and `gradient_cosine_similarity`: these cannot be recovered from the previous scalar CSV fields. Only these three new raw columns are needed; prediction errors and outcome classes are derived offline. The default output is the input directory's sibling `analysis/` directory; the explicit path above makes the destination clear. Keep different run configurations in separate data directories. Reusing an output directory overwrites generated outputs.

For a quick pipeline check without overwriting the full experiment, use separate smoke-run directories:

```bash
smoke_dir=$(mktemp -d)
uv run --locked policy-lab --methods qac_full --steps 600 --eval-every 100 \
  --seeds 0 1 --out-dir "$smoke_dir/data"
uv run --locked policy-lab-plot "$smoke_dir/data" --out-dir "$smoke_dir/analysis"
```

This short run checks the pipeline, not the research conclusions.

### Outputs

| File | Contents |
|---|---|
| `data/config.json` | Experimental configuration |
| `data/source_revision.txt` | Git revision used for the run |
| `data/checkpoints.csv` | Training return and critic-quality measurements |
| `data/counterfactual_updates.csv` | Candidate update measurements, gradient cosine and both gradient norms |
| `analysis/learning_curves.png` | Exact training return, mean ± sample SD across seeds, with optimal-return reference |
| `analysis/critic_quality_over_training.png` | RMSE, action-ranking accuracy, and selection regret over environment steps |
| `analysis/gradient_alignment_over_training.png` | Full-action gradient cosine over environment steps and fraction with defined cosine |
| `analysis/prediction_error_over_training.png` | Signed and absolute learned-guided local-gain errors over environment steps |
| `analysis/predicted_vs_true_gain.png` | Learned-guided predicted versus true local gain, with equality line |
| `analysis/learned_vs_oracle_gain.png` | Paired learned/oracle global gain at checkpoints and labeled seed means |
| `analysis/harmful_update_frequency.png` | All three learned-guided outcome frequencies in fixed 500-step bins, mean ± sample SD across seeds (retained filename) |
| `analysis/critic_rmse_vs_prediction_error.png` | Learned critic RMSE versus absolute local-gain prediction error |
| `analysis/update_kl_vs_improvement.png` | Realized KL versus global gain for both guidance sources, and paired seed mean KLs |
| `analysis/summary.csv` | Per-seed statistics and equally weighted across-seed means/sample SDs |

`summary.csv` has a `seed` column and a `statistic` column (`seed`, `mean`, or `std`); across-seed rows have no seed identifier. Its metric columns are:

- `final_expected_return` and `candidate_count`;
- `mean_signed_local_gain_prediction_error` and `mean_absolute_local_gain_prediction_error`;
- `{beneficial,harmful,effectively_zero}_update_count` and corresponding `_update_frequency` columns;
- `mean_{learned,oracle}_guided_performance_gain` and `mean_paired_oracle_minus_learned_performance_gain`;
- `mean_{learned,oracle}_guided_state_kl` and `mean_paired_oracle_minus_learned_state_kl`;
- `mean_gradient_cosine_similarity`, `defined_gradient_alignment_count`, and `{learned,oracle}_zero_gradient_count`.

Across-seed rows report mean/SD of seed-level counts as well as means/frequencies; they do not pool checkpoint counts across seeds. Seeds with no defined cosine are excluded only from cosine mean/SD, not from other metrics. Gain measures update quality while KL measures update magnitude; identical nominal learning rates do not match the realized KLs.

For implementation tests and repository-wide verification, see [Setup and verification](../../README.md#setup-and-verification).

## Results

## Interpretation


## Limitations

This experiment examines one tabular learner operating on one handcrafted MDP. Its conclusions do not directly generalize to neural function approximation or larger environments.

Critic-quality measurements are uniformly weighted, while candidate updates occur at states visited along training trajectories. Consequently, aggregate critic quality may not represent estimation quality at the states most relevant to learning.

Learned and oracle candidates use identical nominal learning rates but can have different realized KL divergences. The experiment therefore does not independently isolate update direction from update magnitude.

Repeated checkpoints are statistically dependent, and the experiment does not establish causal relationships between critic quality and update performance.

Controlled critic corruption, matched-KL comparisons, and additional environments are outside Experiment 001's scope.
