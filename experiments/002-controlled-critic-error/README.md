# 002 — Controlled Critic Error

## Motivation

Experiment 001 observed reliable policy improvements despite substantial critic value error in a small tabular MDP. It did not identify the errors that cause harmful updates. This experiment uses controlled perturbations to the critic to separate errors in absolute values from errors in action preferences and update direction, allowing their consequences for policy improvement to be examined at fixed policies.

## Research Question

**How do the type and magnitude of critic estimation error affect the reliability of tabular full-action QAC policy improvements?**

## Methodology

### Overview

1. **Retain the baseline.** Use Experiment 001's MDP, tabular QAC learner, and exact policy evaluation framework.
2. **Freeze checkpoints.** Save policy logits, learned critic tables, and the next decision state at regular training checkpoints.
3. **Control critic error.** Construct four synthetic guidance conditions at each frozen policy, using exact action values as the reference.
4. **Compare candidate updates.** Apply the same single-state actor update independently under each condition and evaluate its exact performance change.
5. **Analyze reliability.** Relate perturbation magnitude, gradient alignment, and critic RMSE to harmful update frequency and actual policy improvement.

### 1. Experimental Environment

The experiment retains `branching_mdp_v1`.

| Property | Configuration |
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

Training retains tabular **full-action Q Actor–Critic (QAC)** with an on-policy one-step Expected SARSA critic. Candidate actor updates use the existing softmax-logit update at one decision state:

$$
\theta'_{s,a} = \theta_{s,a} + \alpha \pi(a\mid s)
\left[\widetilde Q(s,a) - \sum_b \pi(b\mid s)\widetilde Q(s,b)\right].
$$

| Parameter | Configuration |
|---|---|
| Training method | `qac_full` |
| Policy / critic representation | Tabular softmax logits / tabular action values |
| Critic update | One-step Expected SARSA |
| Actor / critic learning rate | 0.20 / 0.12 |
| Initialization | Zero actor logits and critic values |
| Training duration | 5,000 environment transitions per seed |
| Checkpoints | Initialization and every 100 completed transitions |
| Training seeds | 0–9 |

### 3. Controlled Critic Perturbations

At each checkpoint, we calculate $Q^\pi$ for the frozen policy and construct the following guidance tables over decision states. These oracle-derived controls isolate imposed errors without confounding them with the learned critic's existing error.

| Condition | Guidance $\widetilde Q(s,a)$ | Purpose |
|---|---|---|
| Oracle | $Q^\pi(s,a)$ | Error-free reference for the same actor update |
| Additive bias | $Q^\pi(s,a) + b$ | Change absolute value error while preserving action preferences |
| Gaussian noise | $Q^\pi(s,a) + \sigma Z_{s,a}$, with independent $Z_{s,a}\sim\mathcal N(0,1)$ | Vary error magnitude and disrupt action preferences |
| Deliberately reversed guidance | $-Q^\pi(s,a)$ | Reverse all non-tied action preferences and the local QAC gradient |

We use a uniform, action-independent bias $b$ as a structural control: its contribution cancels in the centered QAC update, even as RMSE increases. Reversed guidance negates the oracle gradient wherever that gradient is nonzero; its finite-step performance effect is measured rather than assumed.

### 4. Measurements

Each candidate is evaluated against its unchanged checkpoint policy using the existing exact policy evaluation framework.

| Primary measurement | Definition |
|---|---|
| Critic RMSE | Root mean squared error between $\widetilde Q$ and $Q^\pi$, uniformly over decision-state/action pairs |
| Action-ranking accuracy | Fraction of action pairs correctly ordered by $\widetilde Q$, excluding exact-value gaps at or below $10^{-6}$ |
| Gradient alignment | Cosine between perturbed-guidance and oracle full-action logit gradients at the updated state; both gradient norms are also recorded |
| Exact policy improvement | $\Delta J = J(\pi') - J(\pi)$, using exact evaluation of each candidate policy |
| Harmful update frequency | Fraction of candidates with $\Delta J < -10^{-12}$ within each condition and magnitude; beneficial and effectively zero counts are also retained |
| Policy KL divergence | Realized $D_{\mathrm{KL}}(\pi'(\cdot\mid s)\Vert\pi(\cdot\mid s))$ at the updated state |

Records in `data/` identify the source revision, run configuration, training seed, checkpoint step, updated state, condition, magnitude, and noise seed/replicate. Frozen policy and critic tables are saved so paired comparisons can be reconstructed. Summaries and figures are written to `analysis/`.

### 5. Analysis

| Analysis | Purpose |
|---|---|
| Noise magnitude versus harmful update frequency | Measure how reliability varies with Gaussian noise standard deviation, including the zero-noise reference |
| Gradient alignment versus actual policy improvement | Examine whether local update direction predicts the sign and magnitude of exact $\Delta J$ |
| Critic RMSE versus actual policy improvement | Compare error magnitude with exact $\Delta J$ across conditions, including the action-independent bias control |

We summarize noise replicates within checkpoints and checkpoints within each training seed before reporting equally weighted across-seed summaries and variability. Repeated checkpoints and noise draws are not treated as independent training runs. Analyses retain condition, magnitude, training stage, and realized KL so differences in update size and checkpoint policy can be assessed alongside gradient direction and value error. Magnitudes are not rescaled using observed performance outcomes.

## Limitations

The study examines one handcrafted MDP and a tabular learner; it does not establish generalization to function approximation or larger environments. Synthetic errors around exact values isolate controlled interventions but do not reproduce every structure of learned critic error. Single-step counterfactual improvements do not establish long-term learning performance. Finite-step gains depend on policy, state visitation, and update size as well as critic error; gradient alignment alone does not determine their magnitude.
