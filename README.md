# Policy Improvement Lab

**Research question:** How should an actor use imperfect critic-derived action values to make efficient, stable, and appropriately conservative policy improvements?

A small, reproducible research-engineering repository. The first milestone is a **tabular, exactly evaluable** MDP, not a new RL algorithm or a general-purpose framework. It compares online one-step actor–critic (A2C-style), sampled-action QAC, full-action QAC, and a KL-backtracked full-action QAC baseline. Later milestones can add neural critics and Gymnasium tasks.

## What is implemented

- A stochastic finite discounted MDP with **five decision states, three actions, and an absorbing terminal state**.
- Exact policy evaluation with the Bellman equations, providing reference \(V^\pi\), \(Q^\pi\), and performance \(J(\pi)\).
- One-step online actor–critic (`ac_td0`) with a tabular state-value critic. This is **not** yet a batched synchronous A2C reproduction.
- QAC with a tabular action-value critic trained with on-policy one-step **Expected SARSA**.
- Three QAC actor operators: sampled-action gradient, exact action expectation, and full-action gradient with per-state KL backtracking.
- Same-checkpoint counterfactual actor updates using a **frozen learned critic** or an **oracle/exact critic for diagnosis only**.
- Critic RMSE, action-ranking accuracy, action-selection regret, predicted local gain, true local gain, actual return improvement, and realized KL.
- Seeded experiments, CSV outputs, tests, and GitHub Actions CI.

**Important scope:** The KL operator is ordinary backtracking along the full-action gradient. It is *not* a full TRPO/MPO implementation; the initial goal is to isolate step-size effects. The reference Q-function is **never used by the actual learners**.

## Quick start

Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run policy-lab --steps 5000 --eval-every 100 --seeds 0 1 2 --out-dir artifacts/first-run
```

The module entry point is also available:

```bash
uv run python -m policy_improvement_lab --steps 500 --seeds 0 --out-dir artifacts/smoke
```

To produce three diagnostic figures:

```bash
uv sync --locked --extra dev --extra analysis
uv run policy-lab-plot artifacts/first-run
```

`uv.lock` pins dependencies for reproducible setup and CI. Use `uv lock` after
intentional dependency changes and include the updated lockfile. The computational
core requires only NumPy; Pandas, Seaborn, and Matplotlib are optional analysis dependencies.

Plotting conventions: readable method names and fixed method colors across figures,
9 × 5 inch learning curves and 7 × 6 inch scatter plots, exported as PNG at 160 dpi.
Learning curves show the mean exact expected discounted return with shaded ±1 sample
standard deviation (`ddof=1`) across seeds at each environment step; a single seed has
zero-width shading. Scatter plots show individual learned-critic candidate updates,
without aggregation or uncertainty bands. Axis labels identify the measured quantities.

Outputs (gitignored):

- `artifacts/first-run/config.json`: settings, methods, seed list, environment version.
- `artifacts/first-run/checkpoints.csv`: exact expected return and critic diagnostics at each checkpoint.
- `artifacts/first-run/counterfactual_updates.csv`: six candidate updates from each QAC checkpoint (three operators × learned/oracle critic source).
- `artifacts/first-run/*.png`: three diagnostic figures, when plotting is requested.

Read [`docs/experiment_protocol.md`](docs/experiment_protocol.md) before interpreting results, especially the distinction between local and global improvement.

## Research goals, not claims

1. Characterize whether an imperfect learned critic mispredicts the effects of actor updates.
2. Compare the variance and reliability of sampled-action and full-action policy improvements.
3. Investigate whether restricting actor-update size reduces negative effects of critic approximation errors, and at what efficiency cost.
4. Test controlled critic corruption before designing new methods for estimating critic reliability.

A method doing better on this one handcrafted MDP would **not** constitute evidence of generality or algorithmic novelty. See [`docs/roadmap.md`](docs/roadmap.md) for the staged plan.

## Design decisions

- **NumPy-first** for the exact finite-MDP reference experiment. This is intentional: it makes the mathematical tests transparent and cheap. A later phase should introduce PyTorch for function approximation.
- One-action/one-state policy updates permit especially clear diagnostics and testable performance-improvement sign relationships.
- No PVC implementation yet. The original policy–value concordance research motivates the question; PVC can be added as an explicitly controlled baseline later.
- No shared all-purpose RL framework, tracking service, complex configuration system, or cloud infrastructure.

## Prior art to study

- Sutton et al., *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (2000).
- Ciosek & Whiteson, *Expected Policy Gradients* (JMLR 2020).
- Schulman et al., *Trust Region Policy Optimization* (ICML 2015).
- Abdolmaleki et al., *Maximum a Posteriori Policy Optimisation* (2018).
- Fujimoto et al., *Why Should I Trust You, Bellman?* (ICML 2022).

This repository starts as an educational reproduction and analysis project, **not** a claim of a new method.
