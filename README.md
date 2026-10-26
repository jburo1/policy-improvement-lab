# Policy Improvement Lab

During graduate school, I studied actor–critic reinforcement learning and whether a loss comparing the relative ordering of policy and value predictions could improve sample efficiency. Results depended on the configuration and sometimes training failed. This reboot asks a more fundamental question:

**How should an actor use imperfect critic-derived action values to make efficient, stable, and appropriately conservative policy improvements?**

Milestone 0 establishes a tested, exactly evaluable tabular foundation. The first planned experiment follows **one full-action QAC learner** and compares hypothetical full-action improvements guided by its learned critic versus exact values at the same checkpoints. It makes no empirical claims yet.

## Setup and verification

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root:

```sh
uv sync --locked --python 3.11 --extra dev --extra analysis
uv run --locked ruff check .
uv run --locked pytest -q
```

The committed `uv.lock` fixes dependency resolution. CI uses Python 3.11 and uv 0.9.27. Training requires NumPy; the analysis extra supplies Pandas, Seaborn, and Matplotlib. No external RL environment is needed.

A short plumbing check, **not a research experiment**:

```sh
uv run --locked policy-lab --methods qac_full --steps 100 --eval-every 50 --seeds 0 --out-dir artifacts/smoke/data
uv run --locked policy-lab-plot artifacts/smoke/data --out-dir artifacts/smoke/analysis
```

Training writes `checkpoints.csv`, `counterfactual_updates.csv`, and `config.json`. Plotting reads the data directory and writes PNGs to `--out-dir` (defaults to the input directory for compatibility). Commands overwrite matching output files; use a fresh directory per configuration. Keep the source revision and lockfile with any retained research results.

## Experiments

| Experiment | Status | Scope |
| --- | --- | --- |
| [001: Critic reliability](experiments/001-critic-reliability/README.md) | Planned | One full-action QAC learner; paired learned/exact critic diagnostics |

Each experiment contains its question, protocol, commands, results status, and limitations. Generated measurements and configuration belong in its `data/` directory; visualizations and analysis belong in `analysis/`. Algorithms and plotting remain shared in `src/`.

## Implemented foundation

- Five decision states, three actions, and an absorbing, zero-reward terminal state.
- Exact Bellman policy evaluation for reference values and expected discounted return.
- Tabular one-step online actor–critic (`ac_td0`) and Expected-SARSA QAC.
- Sampled-action, full-action, and KL-backtracked full-action QAC operators. Broader comparisons are later work; the backtracked operator is not TRPO or MPO.
- Actor updates read the pre-transition critic. Checkpoint diagnostics compare frozen policy/critic pairs after a stated number of completed transitions.
- Counterfactual candidates never change the learner; exact oracle values are diagnostic only. A separate diagnostic random stream leaves seeded training independent of measurement frequency.
- Critic error, ranking accuracy, selection regret, local gains, exact global gain, and realized KL, with tests of mathematical identities and provenance.

See the [experiment protocol](docs/experiment_protocol.md) for definitions and the [roadmap](docs/roadmap.md) for subsequent work.

## Prior work

- Sutton et al., *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (2000).
- Ciosek & Whiteson, *Expected Policy Gradients* (JMLR 2020).
- Schulman et al., *Trust Region Policy Optimization* (ICML 2015).
- Abdolmaleki et al., *Maximum a Posteriori Policy Optimisation* (2018).
- Fujimoto et al., *Why Should I Trust You, Bellman?* (ICML 2022).
