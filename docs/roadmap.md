# Research roadmap

## Milestone 0 — Current scaffold (implemented)

- [x] Exact finite-MDP policy evaluation with tests.
- [x] Tabular one-step online actor–critic (A2C-style) and Expected-SARSA QAC.
- [x] Sampled QAC, full-action QAC, KL-backtracked full-action QAC.
- [x] Counterfactual paired update diagnostics and multi-seed CSV export.
- [x] CI tests and linting.

**Exit criterion:** tests pass; baseline returns are finite and generally improve over seeded runs; diagnostic sign invariants hold.

## Milestone 1 — Analysis and controlled critic errors

- [ ] Produce four core plots described in `experiment_protocol.md`.
- [ ] Sweep critic error magnitude and *structured* action-dependent bias at frozen checkpoints.
- [ ] Sweep actor learning rates and KL bounds; compare operators at matched realized KL.
- [ ] Run >=10 independent seeds; report effect sizes and dispersion.
- [ ] Add a second independent small MDP, avoiding overfitting conclusions to the original.

**Exit criterion:** explain when the learned critic's predicted update gain overstates or understates actual gain, and identify at least one reproducible failure mode.

## Milestone 2 — Neural function approximation

- [ ] Implement a proper synchronous n-step A2C reproduction with tests.
- [ ] Introduce PyTorch Q-network and actor with tests; keep the exact finite MDP as a reference environment.
- [ ] Preserve frozen-policy evaluation when assessing critic Q-error.
- [ ] Add Gymnasium CartPole, then LunarLander or a discrete benchmark where differences do not saturate immediately.
- [ ] Use sampled Monte Carlo references only where exact Q is unavailable, with uncertainty intervals.

**Exit criterion:** show whether relationships found in the tabular reference carry over to learned neural critics.

## Milestone 3 — Established conservative-improvement baselines

- [ ] Implement a correctly scoped KL-constrained target-policy fit or reproduce MPO/another suitable baseline.
- [ ] Compare under shared critic states and training budgets; measure wall-clock cost.
- [ ] Decide whether a new reliability-adaptive method is justified by an observed gap.

**Exit criterion:** a reproducible technical report with explicit claims, limitations, negative results, and links to raw experiments.

## Out of scope unless results justify it

- Critic ensembles and Bayesian uncertainty estimation.
- SAC, MuZero, and image-based RL.
- Distributed training, cloud orchestration, and generic modular RL infrastructure.
- Claiming the KL-backtracked baseline is equivalent to TRPO or MPO.
