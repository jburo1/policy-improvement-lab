# 001 — Critic Reliability


## Motivation

Actor–critic (AC) reinforcement learning relies on learned value estimates to guide or bootstrap policy improvement. However, during training, these estimates are imperfect, particularly at the beginning when experience is limited and the critic is inaccurate and continually adapting to the changing policy.

Importantly, accurate value estimation and reliable policy improvement are not necessarily equivalent. A critic may exhibit substantial value-prediction error while preserving the relative action preferences needed to improve a policy. Conversely, errors in those preferences may cause the actor to reinforce inferior actions, potentially reducing performance.

This experiment investigates the relationship between **critic estimation quality** and the **reliability of critic-guided policy updates**. By reliability, we mean whether the policy update actually improves the policy.

To this end, we construct a small, exactly evaluable Markov decision process that provides access to the true action values and expected return of any policy. This makes it possible to distinguish errors in the learned critic from their consequences for policy improvement. We start small and, dependent on results, move towards more complicated algorithms.

## Research Question

**How reliably do learned action-value estimates guide beneficial policy improvements, relative to exact action values, in a simple tabular QAC RL training run?**

The investigation focuses on three aspects:

1. **Prediction accuracy:** How accurately does the learned critic predict the improvement produced by its own policy update?
2. **Update reliability:** How frequently do learned-critic-guided updates increase or decrease true policy performance?
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

The finite state space permits exact computation of \(Q^\pi(s,a)\) and expected discounted return \(J(\pi)\) for every checkpoint policy.

### 2. Learning Algorithm

Training uses a tabular **full-action Q Actor–Critic (QAC)** learner.

The critic estimates action values using on-policy one-step Expected SARSA. The actor performs gradient ascent on the expected action value under the current policy:

\[
\theta_{s,a} \leftarrow \theta_{s,a} +
\alpha \pi(a|s)
\left[
\hat Q(s,a) -
\sum_b \pi(b|s)\hat Q(s,b)
\right].
\]

Both actor and critic updates use the policy and critic estimates from before the current transition's learning updates.

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

Checkpoints include step 0 and the final training step. Checkpoint \(k\) represents the actor and critic after \(k\) completed transitions and learning updates.

### 3. Counterfactual Policy Updates

At each checkpoint, the current policy and learned critic are frozen.

Two hypothetical full-action QAC updates are constructed at the checkpoint's next decision state.

| Condition | Learned guidance | Oracle guidance |
|---|---|---|
| Starting policy | Current checkpoint policy | Identical |
| Update operator | Full-action QAC | Full-action QAC |
| Critic guidance | Learned \(\hat Q\) | Exact \(Q^\pi\) |
| Actor learning rate | 0.20 | 0.20 |
| Updated states | One | Same state |
| Training impact | None | None |

Both candidates are evaluated against the same starting policy. Neither candidate is committed to training.

This paired comparison isolates the difference between learned and exact critic guidance at a fixed checkpoint, although their resulting update magnitudes may differ.

The exact critic is a diagnostic reference, not a separately trained learner or a guarantee of optimal finite-step improvement.

### 4. Measurements

Three categories of measurements characterize critic quality and policy-improvement reliability.

**Critic quality**

| Metric | Description |
|---|---|
| RMSE | Error between learned and exact action values |
| Action-ranking accuracy | Fraction of eligible action pairs correctly ordered by the learned critic |
| Action-selection regret | True value lost by selecting the learned critic's preferred action rather than the optimal action |

Critic-quality measurements are calculated uniformly over decision states and actions rather than weighted by visitation frequency.

**Candidate update quality**

| Metric | Description |
|---|---|
| Predicted local gain | Estimated improvement under the guidance critic |
| True local gain | Local improvement evaluated using exact action values |
| Actual performance gain | Change in exact expected return, \(J(\pi')-J(\pi)\) |
| Realized state KL | \(D_{\mathrm{KL}}(\pi'(\cdot\mid s)\|\pi(\cdot\mid s))\) |

**Training performance**

| Metric | Description |
|---|---|
| Expected return | Exact discounted return of the trained policy at each checkpoint |
| Training progress | Number of completed environment transitions |

All measurements follow the definitions in the [shared experimental protocol](../../docs/experiment_protocol.md).

### 5. Analysis

The primary analysis compares learned-guided and oracle-guided candidate updates at matched checkpoints.

| Analysis | Purpose |
|---|---|
| Predicted versus true local gain | Characterize the accuracy of the learned critic's improvement predictions |
| Negative performance gain frequency | Quantify the occurrence of detrimental learned-guided updates |
| Paired learned/oracle performance gain | Measure differences in realized improvement under learned versus exact guidance |
| Critic quality versus update error | Examine relationships between critic estimation quality and improvement reliability |
| Expected return over training | Contextualize the evolution of the learner |

A performance gain below `-1e-12` is classified as negative to avoid treating numerical noise as a performance decrease.

Learned and oracle candidates are paired using their originating training method, seed, checkpoint step, state, action, and candidate operator. Only `qac_full` training runs and `qac_full` candidates are included in the primary analysis.

Measurements are summarized within each seed before computing across-seed means and sample standard deviations. Checkpoints are repeated observations from the same training trajectory and are not treated as independent experimental replicates.

Associations between critic quality and update performance are descriptive rather than causal.

## Reproduction

The following commands reproduce the experiment's raw measurements and currently implemented visualizations.

**Requirements:** Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

Run from the repository root.

### 1. Install dependencies

```bash
uv sync --locked --python 3.11 --extra analysis
```

### 2. Run the experiment

```bash
git rev-parse HEAD > experiments/001-critic-reliability/data/source_revision.txt

uv run --locked policy-lab \
  --methods qac_full \
  --steps 5000 \
  --eval-every 100 \
  --actor-lr 0.20 \
  --critic-lr 0.12 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --out-dir experiments/001-critic-reliability/data
```

The training command records checkpoint metrics and counterfactual update measurements.

### 3. Generate visualizations

```bash
uv run --locked policy-lab-plot \
  experiments/001-critic-reliability/data \
  --out-dir experiments/001-critic-reliability/analysis
```

### Outputs

| File | Contents |
|---|---|
| `data/config.json` | Experimental configuration |
| `data/source_revision.txt` | Git revision used for the run |
| `data/checkpoints.csv` | Training return and critic-quality measurements |
| `data/counterfactual_updates.csv` | Candidate update measurements |
| `analysis/learning_curves.png` | Training performance |
| `analysis/learned_vs_oracle_gain.png` | Paired learned/oracle comparison |

The generic plotting command also generates `predicted_vs_true_gain.png` and `update_kl_vs_improvement.png`. Those figures include additional diagnostic operators emitted by the shared exporter and are not restricted to Experiment 001's primary comparison.

The complete seed-level statistical analysis is not yet automated. The commands above reproduce the underlying measurements and existing figures, rather than a finalized set of statistical results.

For implementation tests and repository-wide verification, see [Setup and verification](../../README.md#setup-and-verification).

**Reproducibility:** The recorded revision identifies the committed source code. Execute the experiment from a clean working tree, preserve the corresponding `uv.lock`, and retain the complete configuration and all seed results. Reusing the output directory overwrites existing CSVs and figures.

## Results

**Analysis in progress.**

No completed experimental results are currently committed to the repository.

Results and visualizations will be included here as the analysis is completed.

## Interpretation

Interpretation is reserved for the observed measurements.

In particular, differences between learned-guided and oracle-guided updates must be considered alongside their realized KL divergence, since identical actor learning rates do not guarantee identical update magnitudes.

For these single-state updates, the exact performance-difference identity connects true local gain with global performance change. The relationship is an implementation invariant, not an independent empirical finding.

## Limitations

This experiment examines one tabular learner operating on one handcrafted MDP. Its conclusions do not directly generalize to neural function approximation or larger environments.

Critic-quality measurements are uniformly weighted, while candidate updates occur at states visited along training trajectories. Consequently, aggregate critic quality may not represent estimation quality at the states most relevant to learning.

Learned and oracle candidates use identical nominal learning rates but can have different realized KL divergences. The experiment therefore does not independently isolate update direction from update magnitude.

Repeated checkpoints are statistically dependent, and the experiment does not establish causal relationships between critic quality and update performance.

Controlled critic corruption, matched-KL comparisons, and additional environments are outside Experiment 001's scope.
