# 002 — Controlled Critic Error

## Motivation

Experiment 001 observed reliable policy improvements despite substantial critic value error in a small tabular MDP. It did not identify the errors that cause harmful updates. Controlled perturbations will separate errors in absolute values from errors in action preferences and update direction, allowing their consequences for policy improvement to be examined at fixed policies.

## Research Question

**How do the type and magnitude of critic estimation error affect the reliability of tabular full-action QAC policy improvements?**

## Methodology

### Overview

1. **Retain the baseline.** We will use Experiment 001's MDP, tabular QAC learner, and exact policy evaluation framework.
2. **Freeze checkpoints.** We will save policy logits, learned critic tables, and the next decision state at regular training checkpoints.
3. **Control critic error.** We will construct four synthetic guidance conditions at each frozen policy, using exact action values as the reference.
4. **Compare candidate updates.** We will apply the same single-state actor update independently under each condition and evaluate its exact performance change.
5. **Analyze reliability.** We will relate perturbation magnitude, gradient alignment, and critic RMSE to harmful update frequency and actual policy improvement.

### 1. Experimental Environment

The experiment will retain `branching_mdp_v1` without changing its transitions or rewards.

| Property | Planned configuration |
|---|---|
| Environment | `branching_mdp_v1` |
| Decision states | 5 |
| Actions per state | 3 |
| Terminal states | 1 absorbing state |
| Discount factor | 0.95 |
| Initial state | State 0 |
| Reward structure | Transition costs and delayed terminal reward |
| Reference evaluation | Exact Bellman solution for $Q^\pi$ and expected discounted return $J(\pi)$ |

### 2. Learning Algorithm

Training will retain tabular **full-action Q Actor–Critic (QAC)** with an on-policy one-step Expected SARSA critic. Candidate actor updates will use the existing softmax-logit update at one decision state:

$$
\theta'_{s,a} = \theta_{s,a} + \alpha \pi(a\mid s)
\left[\widetilde Q(s,a) - \sum_b \pi(b\mid s)\widetilde Q(s,b)\right].
$$

| Parameter | Planned configuration |
|---|---|
| Training method | `qac_full` |
| Policy / critic representation | Tabular softmax logits / tabular action values |
| Critic update | One-step Expected SARSA |
| Actor / critic learning rate | 0.20 / 0.12 |
| Initialization | Zero actor logits and critic values |
| Training duration | 5,000 environment transitions per seed |
| Checkpoints | Initialization and every 100 completed transitions |
| Training seeds | 0–9 |

Checkpoint $k$ will describe the policy and critic **after** both learning updates for transition $k$ and **before** the next transition; checkpoint zero will precede learning. All candidate updates will start from copies of that same frozen policy, change only the checkpoint's next decision state, and use actor learning rate 0.20. Candidates will not be chained or fed back into training, and the learned critic will remain frozen throughout the comparison.

Exact values will be used only for evaluation and the explicitly oracle-derived synthetic controls below. The training learner will receive neither exact values nor perturbed guidance. Diagnostic random streams will remain separate from training randomness.

### 3. Controlled Critic Perturbations

At each checkpoint, we will calculate $Q^\pi$ for the frozen policy and construct the following guidance tables over decision states. These oracle-derived controls will isolate imposed errors without confounding them with the learned critic's existing error. The frozen learned critic will be retained with the checkpoint for provenance.

| Condition | Planned guidance $\widetilde Q(s,a)$ | Purpose |
|---|---|---|
| Oracle | $Q^\pi(s,a)$ | Error-free reference for the same actor update |
| Additive bias | $Q^\pi(s,a) + b$ | Change absolute value error while preserving action preferences |
| Gaussian noise | $Q^\pi(s,a) + \sigma Z_{s,a}$, with independent $Z_{s,a}\sim\mathcal N(0,1)$ | Vary error magnitude and disrupt action preferences |
| Deliberately reversed guidance | $-Q^\pi(s,a)$ | Reverse all non-tied action preferences and the local QAC gradient |

We will use a uniform, action-independent bias $b$ as a structural control: its contribution will cancel in the centered QAC update, even as RMSE increases. Reversed guidance will negate the oracle gradient wherever that gradient is nonzero; its finite-step performance effect will still be measured rather than assumed.

We will specify signed bias levels, nonnegative noise standard deviations (including zero), and the number of independent noise draws per checkpoint before running the experiment. Magnitudes will be expressed in reward-value units and saved in the run configuration. For paired comparisons across noise magnitudes, each noise replicate will reuse the same standard-normal table $Z$ scaled by $\sigma$. Terminal values will remain zero and will not be perturbed or included in critic metrics.

### 4. Measurements

Each candidate will be evaluated against its unchanged checkpoint policy using the existing exact policy evaluation framework.

| Primary measurement | Planned definition |
|---|---|
| Critic RMSE | Root mean squared error between $\widetilde Q$ and $Q^\pi$, uniformly over decision-state/action pairs |
| Action-ranking accuracy | Fraction of action pairs correctly ordered by $\widetilde Q$, excluding exact-value gaps at or below $10^{-6}$ |
| Gradient alignment | Cosine between perturbed-guidance and oracle full-action logit gradients at the updated state; both gradient norms will also be recorded |
| Exact policy improvement | $\Delta J = J(\pi') - J(\pi)$, using exact evaluation of each candidate policy |
| Harmful update frequency | Fraction of candidates with $\Delta J < -10^{-12}$ within each condition and magnitude; beneficial and effectively zero counts will also be retained |
| Policy KL divergence | Realized $D_{\mathrm{KL}}(\pi'(\cdot\mid s)\Vert\pi(\cdot\mid s))$ at the updated state |

Gradient cosines will be undefined when either gradient norm is at or below $10^{-12}$; undefined values will be counted and excluded from alignment summaries. Ranking accuracy will be undefined if no eligible action pairs exist. These cases will not be interpreted as zero alignment or zero accuracy.

Records in `data/` will identify the source revision, run configuration, training seed, checkpoint step, updated state, condition, magnitude, and noise seed/replicate. Frozen policy and critic tables will be saved so paired comparisons can be reconstructed. Summaries and figures will be written to `analysis/`.

### 5. Analysis

The planned analyses will compare candidates from matched frozen checkpoints, with oracle guidance as the reference.

| Planned analysis | Purpose |
|---|---|
| Noise magnitude versus harmful update frequency | Measure how reliability varies with Gaussian noise standard deviation, including the zero-noise reference |
| Gradient alignment versus actual policy improvement | Examine whether local update direction predicts the sign and magnitude of exact $\Delta J$ |
| Critic RMSE versus actual policy improvement | Compare error magnitude with exact $\Delta J$ across conditions, including the action-independent bias control |

We will summarize noise replicates within checkpoints and checkpoints within each training seed before reporting equally weighted across-seed summaries and variability. Repeated checkpoints and noise draws will not be treated as independent training runs. Analyses will retain condition, magnitude, training stage, and realized KL so differences in update size and checkpoint policy can be assessed alongside gradient direction and value error. Magnitudes will not be rescaled using observed performance outcomes.

## Results

*Placeholder: results, figures, and statistical summaries will be added after implementation and execution.*

## Interpretation

*Placeholder: interpretation will be added after results are available.*

## Limitations

The planned study will examine one handcrafted MDP and a tabular learner; it will not establish generalization to function approximation or larger environments. Synthetic errors around exact values will isolate controlled interventions but will not reproduce every structure of learned critic error. Single-step counterfactual improvements will not establish long-term learning performance. Finite-step gains will depend on policy, state visitation, and update size as well as critic error; gradient alignment alone will not determine their magnitude.

## Reproduction

This directory currently contains documentation and empty output directories only. Commands and the remaining perturbation settings will be defined during implementation.

### 1. Run the experiment

```bash
# TODO: define the command to train/save frozen checkpoints and evaluate
# controlled critic conditions, writing records and configuration to
# experiments/002-controlled-critic-error/data/.
```

### 2. Generate figures and statistical summaries

```bash
# TODO: define the command to analyze saved candidate measurements and
# write figures and summaries to experiments/002-controlled-critic-error/analysis/.
```
