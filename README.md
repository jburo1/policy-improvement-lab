# Policy Improvement Lab

During graduate school, I studied actor–critic reinforcement learning and whether a loss comparing the relative ordering of policy and value predictions could improve sample efficiency on benchmark RL envs. This reboot lives in the same conceptual space but asks a more fundamental question:

**How should an actor use imperfect critic-derived action values to make efficient, stable, and appropriately conservative policy improvements?**

The following experiments investigate this.

The research progression I have in mind is the following:

First: Establish whether and when critic errors lead to poor policy improvements.

Second: Understand which aspects of critic error are responsible.

Third: Investigate whether we can design actor updates that are more robust to those errors.

## Experiments

| Experiment | Status | Scope |
| --- | --- | --- |
| [001: Critic reliability](experiments/001-critic-reliability/README.md) | Planned | One full-action QAC learner; paired learned/exact critic diagnostics |


## Setup and verification

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root:

```sh
uv sync --locked --python 3.11 --extra dev --extra analysis
uv run --locked ruff check .
uv run --locked pytest -q
```


## Prior work

- Sutton et al., *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (2000).
- Ciosek & Whiteson, *Expected Policy Gradients* (JMLR 2020).
- Schulman et al., *Trust Region Policy Optimization* (ICML 2015).
- Abdolmaleki et al., *Maximum a Posteriori Policy Optimisation* (2018).
- Fujimoto et al., *Why Should I Trust You, Bellman?* (ICML 2022).
