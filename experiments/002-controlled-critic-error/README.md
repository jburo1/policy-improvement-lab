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

Training calls the same `train(..., "qac_full", ...)` implementation as EXP001.
Checkpoints describe parameters **after** the specified number of completed
transitions and their learning updates; step 0 precedes learning. The frozen
state is the next decision state, including a reset to state 0 after termination.
The learner's training RNG and original diagnostic RNG are unchanged. An optional
checkpoint callback receives copies of the logits and learned Q table. Synthetic
guidance is used only for uncommitted counterfactuals and never feeds back into
training. Every candidate starts from the same frozen logits and updates only
the same state's row, with actor rate 0.20 and no KL backtracking.

### 3. Controlled Critic Perturbations

At each checkpoint, we calculate $Q^\pi$ for the frozen policy and construct the following guidance tables over decision states. These oracle-derived controls isolate imposed errors without confounding them with the learned critic's existing error.

| Condition | Guidance $\widetilde Q(s,a)$ | Purpose |
|---|---|---|
| Oracle | $Q^\pi(s,a)$ | Error-free reference for the same actor update |
| Additive bias | $Q^\pi(s,a) + b$ | Change absolute value error while preserving action preferences |
| Gaussian noise | $Q^\pi(s,a) + \sigma Z_{s,a}$, with independent $Z_{s,a}\sim\mathcal N(0,1)$ | Vary error magnitude and disrupt action preferences |
| Deliberately reversed guidance | $-Q^\pi(s,a)$ | Reverse all non-tied action preferences and the local QAC gradient |

We use a uniform, action-independent bias $b$ as a structural control: its contribution cancels in the centered QAC update, even as RMSE increases. Reversed guidance negates the oracle gradient wherever that gradient is nonzero; its finite-step performance effect is measured rather than assumed.

#### Finalized magnitude grid and sampling

The additive-bias grid is **$b\in\{0,1,10\}$**, in action-value units. Oracle and
reversed guidance each have one candidate per checkpoint. For storage, their
`magnitude` labels are 0 and 1 respectively; these labels are not additional sweeps.
Terminal-state values remain unchanged under every condition.

For Gaussian noise, define a gap scale separately at every decision state:

$$
g_s=\max\left(10^{-6},
\sqrt{\frac{1}{\binom{3}{2}}\sum_{a<b}
\left[Q^\pi(s,a)-Q^\pi(s,b)\right]^2}\right),
\qquad \sigma_s=m g_s.
$$

Use **$m\in\{0,0.25,0.5,1,2,4,8\}$**, with **20 independent table draws at
every checkpoint and magnitude**, including zero. Each draw has independent
standard-normal entries over decision states/actions; its standard deviations
are state-specific. A single multiplier therefore need not imply a single
absolute standard deviation over the table. Record `local_gap_scale` and
`local_noise_sigma` at the updated state. This grid is fixed before execution
and does not depend on observed policy improvements.

If all pairwise gaps vanish or the RMS gap is below $10^{-6}$, use the floor
$g_s=10^{-6}$ rather than dividing by zero, omitting the checkpoint, or silently
removing noise. Ranking accuracy excludes exact-value pair gaps at or below
$10^{-6}$ independently of this scaling rule; no eligible pairs yields NaN.
Gradient cosine is NaN when either norm is at most $10^{-12}$. Summaries exclude
undefined values from their means and retain defined counts/frequencies.

The default noise root is **0**. Each sample seed is the uint64 output of
`SeedSequence([2002, noise_root, training_seed, checkpoint_step, low32(m_bits),
high32(m_bits), replicate]).generate_state(1, dtype=uint64)`, where `m_bits` is
the IEEE-754 float64 encoding of the multiplier. `default_rng(sample_seed)`
draws a `(5, 3)` normal table in decision-state order. The seed is saved as a
hexadecimal string to avoid CSV integer precision loss. Keys do not depend on
iteration order; noise never consumes training randomness. Replicates are 0–19;
deterministic controls use replicate −1 and an empty noise seed. Software
versions are recorded for reproducibility.

The complete configuration produces **510 frozen checkpoints** (51 per seed)
and **73,950 candidate rows** (145 per checkpoint).

### 4. Measurements

Each candidate is evaluated against its unchanged checkpoint policy using the existing exact policy evaluation framework.

| Primary measurement | Definition |
|---|---|
| Critic RMSE | Root mean squared error between $\widetilde Q$ and $Q^\pi$, uniformly over decision-state/action pairs |
| State-local critic RMSE | Same RMSE restricted to the updated state's three actions (`local_rmse`) |
| Action-ranking accuracy | Fraction of action pairs correctly ordered by $\widetilde Q$, excluding exact-value gaps at or below $10^{-6}$ |
| State-local action-ranking accuracy | Same eligible-pair fraction at the updated state (`local_ranking_accuracy`); a predicted tie is incorrect |
| Gradient alignment | Cosine between perturbed-guidance and oracle full-action logit gradients at the updated state; both gradient norms are also recorded |
| Exact policy improvement | $\Delta J = J(\pi') - J(\pi)$, using exact evaluation of each candidate policy |
| Exact local policy improvement | $\Delta_s=\sum_a[\pi'(a\mid s)-\pi(a\mid s)]Q^\pi(s,a)$ (`true_local_gain`) |
| Paired performance difference | Oracle $\Delta J$ minus this candidate's $\Delta J$ at the identical frozen checkpoint (`oracle_minus_perturbed_gain`); positive favors oracle |
| Harmful update frequency | Fraction of candidates with $\Delta J < -10^{-12}$ within each condition and magnitude; beneficial and effectively zero counts are also retained |
| Policy KL divergence | Realized $D_{\mathrm{KL}}(\pi'(\cdot\mid s)\Vert\pi(\cdot\mid s))$ at the updated state |

The raw data also retains predicted local gain under the candidate's guidance,
global action-selection regret, checkpoint expected return, and the matched
oracle gain. Critic-quality fields on EXP002 rows describe that row's synthetic
guidance (EXP001's learned-critic field semantics are unchanged). For single-state
updates, exact global gain equals exact local gain times the new policy's
unnormalized discounted occupancy of the updated state; local and global gains
are distinct quantities.

Records in `data/` identify the source revision, run configuration, training seed, checkpoint step, updated state, condition, magnitude, and noise seed/replicate. Frozen policy and critic tables are saved so paired comparisons can be reconstructed. Summaries and figures are written to `analysis/`.

### 5. Analysis

| Analysis | Purpose |
|---|---|
| Noise magnitude versus harmful update frequency | Measure how reliability varies with Gaussian noise standard deviation, including the zero-noise reference |
| Gradient alignment versus actual policy improvement | Examine whether local update direction predicts the sign and magnitude of exact $\Delta J$ |
| Critic RMSE versus actual policy improvement | Compare error magnitude with exact $\Delta J$ across conditions, including the action-independent bias control |

We summarize noise replicates within checkpoints and checkpoints within each training seed before reporting equally weighted across-seed summaries and variability. Repeated checkpoints and noise draws are not treated as independent training runs. Analyses retain condition, magnitude, training stage, and realized KL so differences in update size and checkpoint policy can be assessed alongside gradient direction and value error. Magnitudes are not rescaled using observed performance outcomes.

The analysis validates the complete configured candidate grid, unique
checkpoint/replicate keys, checkpoint returns, and paired oracle differences
before aggregating. Pairing uses `(run_id, seed, step, state)`. No merge or
aggregation combines different run configurations. Each noise checkpoint's
harmful frequency is the fraction of its 20 draws that are harmful; it is not
the sign of their mean gain. All checkpoints, including initialization, have
equal weight within each seed. Across-seed variability is the **sample standard
deviation**, not a confidence interval. A single-seed SD remains NaN in CSVs;
figure bands have zero width for singletons. Every metric's across-seed summary
includes its contributing-seed count, including metrics with undefined values.

Training-stage summaries use fixed bins `[0, 500]`, `(500, 1000]`, …,
`(4500, 5000]`, with the same draw → checkpoint → seed aggregation within bins.
Outcome and defined-gradient counts sum actual candidates; frequency and gain
means retain checkpoint and seed weighting rather than pooling those counts.

### Output files

`data/` contains:

- `config.json`: finalized configuration, run ID, revision, checkout dirty flag,
  source-module SHA-256 hashes, and Python/NumPy versions. The run ID hashes this
  metadata excluding the dirty flag, so generated output files cannot change it.
- `source_revision.txt`: originating Git revision (hashes also identify uncommitted code).
- `checkpoints.csv`: training performance/learned-critic diagnostics, seed, step,
  next decision state, run ID, and revision.
- `frozen_checkpoints.npz`: aligned arrays `seed`, `step`, `state`, `logits`,
  `critic_q`, plus `run_id`. Load with `numpy.load`; recompute exact Q from each
  saved policy and reconstruct draws using the recorded keys.
- `counterfactual_updates.csv`: every candidate, synthetic-guidance metrics,
  outcome indicators, pairing keys, magnitude, replicate, and noise seed.

`analysis/` contains:

- `checkpoint_summary.csv`: draw means, candidate/outcome counts, defined-gradient
  counts/frequencies, and training-stage bin for each checkpoint/condition/magnitude.
- `seed_summary.csv` and `summary.csv`: within-seed checkpoint means/counts,
  followed by separate across-seed mean, sample SD, and contributing-seed count columns.
- `stage_seed_summary.csv` and `stage_summary.csv`: the same hierarchy by training stage.
- `noise_magnitude_vs_harmful_frequency.png`: mean ± seed SD overall and by stage,
  including zero noise.
- `gradient_alignment_vs_improvement.png`: checkpoint-mean alignment versus exact
  global improvement, with realized KL versus improvement alongside it.
- `critic_rmse_vs_improvement.png`: global and state-local RMSE versus exact global improvement.

Scatter plots distinguish conditions and show descriptive checkpoint means;
they are not independent-run observations or inferential correlation tests.
Magnitude, stage, and KL remain in the summary tables for stratified inspection;
raw draws retain within-checkpoint relationships that scatter means can obscure.

## Limitations

The study examines one handcrafted MDP and a tabular learner; it does not establish generalization to function approximation or larger environments. Synthetic errors around exact values isolate controlled interventions but do not reproduce every structure of learned critic error. Single-step counterfactual improvements do not establish long-term learning performance. Finite-step gains depend on policy, state visitation, and update size as well as critic error; gradient alignment alone does not determine their magnitude.

The checkpoint decision state is visited rather than uniformly sampled, while
global critic metrics weight states/actions uniformly. The gap floor provides
an absolute noise scale at near-ties rather than a meaningful relative ratio.
Equal nominal actor rates do not match realized KL across conditions. Twenty
noise draws give a coarse checkpoint harmful-frequency resolution of 0.05;
ten training seeds limit precision of across-seed variability. Finite-step
oracle guidance is a reference, not an assumed best candidate. No full EXP002
results have been generated as part of preparation; smoke outputs validate
execution only.

## Reproduction

Run from the repository root, using the existing locked environment:

```bash
uv sync --locked --python 3.11 --extra dev --extra analysis
```

### Complete experiment (not run during preparation)

```bash
uv run --locked python -m policy_improvement_lab.controlled_error \
  --steps 5000 --eval-every 100 \
  --actor-lr 0.20 --critic-lr 0.12 \
  --seeds 0 1 2 3 4 5 6 7 8 9 --noise-root 0 \
  --out-dir experiments/002-controlled-critic-error/data

uv run --locked python -m policy_improvement_lab.controlled_error_plot \
  experiments/002-controlled-critic-error/data \
  --out-dir experiments/002-controlled-critic-error/analysis
```

These are also the runner's defaults; analysis is a separate offline operation
and never invokes learning. Choose a fresh output directory for a distinct
configuration: files with the same names are replaced when rerunning.

### Minimal smoke run (two seeds, two checkpoints per seed)

Smoke artifacts stay under the repository's ignored `artifacts/` directory:

```bash
uv run --locked python -m policy_improvement_lab.controlled_error \
  --steps 10 --eval-every 10 --seeds 0 1 --noise-root 0 \
  --out-dir artifacts/002-controlled-critic-error/data

uv run --locked python -m policy_improvement_lab.controlled_error_plot \
  artifacts/002-controlled-critic-error/data \
  --out-dir artifacts/002-controlled-critic-error/analysis

uv run --locked pytest -q
uv run --locked ruff check .
```

This smoke run retains the complete condition/magnitude grid and all 20 noise
draws: four frozen checkpoints and 580 candidates. Focused EXP002 tests are in
`tests/test_controlled_error.py`, including perturbations, deterministic draws,
observer/candidate isolation, exact metric identities, hierarchical weighting,
pairing validation, saved-checkpoint reconstruction, and figure generation.
