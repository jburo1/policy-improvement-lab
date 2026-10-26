# Research roadmap

## Milestone 0 — Correct, reproducible foundation

- [x] Inspect and correct actor/critic timing; both updates read pre-transition estimates.
- [x] Exact finite-MDP evaluation with Bellman, analytic-reference, and single-state performance-difference tests.
- [x] Test full-action gradients numerically and sampled/full-action equivalence in expectation.
- [x] Paired counterfactual diagnostics with originating training method, seed, and checkpoint step.
- [x] Checkpoints describe policy and critic after the same number of completed transitions.
- [x] Protect learned/oracle separation and deterministic execution from diagnostic side effects.
- [x] Locked uv setup, CI repair, CSV generation, and separate analysis output.
- [x] Self-contained planned experiment 001 and index.

**Exit criterion:** lint, focused tests, and a short CSV/plot smoke run pass. This establishes implementation invariants, not learning performance or a scientific result.

## Milestone 1 — First critic-reliability experiment (planned)

Follow [experiment 001](../experiments/001-critic-reliability/README.md):

- [ ] Train only full-action QAC on the fixed tabular MDP using learned estimates.
- [ ] At frozen checkpoints, compare full-action learned-guided and oracle-guided candidate updates with identical policy, state, and nominal actor learning rate.
- [ ] Run at least 10 independent seeds with a fixed configuration; retain raw measurements, configuration, source revision, and lockfile.
- [ ] Analyze gain prediction errors, harmful-update frequencies, and paired learned/oracle differences; summarize dispersion across seeds.
- [ ] Record results, interpretation, negative findings, and limitations without selecting configurations for preferred outcomes.

**Exit criterion:** a reproducible descriptive analysis of when this learner's critic correctly or incorrectly predicts update benefit. Finding a failure mode is not required.

## Later tabular work

- Compare sampled and full-action operators and training trajectories; account for different visited data.
- Study actor learning rates, KL caps, and comparisons at matched realized KL.
- Introduce controlled random, action-dependent, and structured critic corruption at frozen checkpoints.
- Add another small MDP to test whether observations depend on the handcrafted environment.

These are separate experiments, not requirements for experiment 001.

## Neural function approximation and established baselines

- Reproduce synchronous n-step A2C and introduce neural actors/critics with tests.
- Retain frozen-policy references; use Monte Carlo estimates with uncertainty where exact values are unavailable.
- Add discrete benchmarks, then investigate whether tabular relationships carry over.
- Reproduce a suitable established conservative-improvement method before considering new methods.
- Compare shared critic states, training budgets, and measured compute costs.

## Out of scope unless evidence justifies it

- Critic ensembles, Bayesian estimation, SAC, MuZero, and image-based RL.
- Distributed training, cloud orchestration, generic experiment frameworks.
- Claiming the KL-backtracked baseline is equivalent to TRPO or MPO.
