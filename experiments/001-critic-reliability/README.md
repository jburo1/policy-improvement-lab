# 001 — Critic reliability

**Status: planned.** Milestone 0 validates the implementation and output plumbing only. No full experiment has been conducted and no research results are reported.

## Research question

How reliably does a learned action-value critic predict the benefit of a full-action policy update along one full-action QAC training trajectory? How do hypothetical learned-guided and exact-guided updates differ at the same checkpoint?

## Methodology

Train `qac_full` on `branching_mdp_v1`: five decision states, three actions, absorbing terminal state, discount 0.95, initial state 0. The actor and Q critic start at zero. Q learning uses on-policy one-step Expected SARSA; the actor uses the frozen pre-transition critic. Exact values never guide committed updates.

Planned configuration: 5,000 transitions per seed, evaluation every 100 transitions, actor rate 0.20, critic rate 0.12, seeds 0–9. Include the initial and final checkpoints. At each checkpoint, freeze the current policy and learned critic and construct two hypothetical full-action updates at the next visited decision state: one using learned Q and one using exact Q of that policy. Both use the same nominal actor rate and change only that state. Neither is committed.

The exporter also retains existing sampled and KL-backtracked candidate diagnostics; filter to `method=qac_full` for this experiment. Only `training_method=qac_full` is trained. Broader operator comparisons, KL sweeps, controlled corruption, and additional MDPs are later work.

## Measurements and planned analysis

Use the definitions and checkpoint conventions in the shared [protocol](../../docs/experiment_protocol.md).

- Track exact expected return, learned critic RMSE, action-pair ranking accuracy, and action-selection regret.
- For both full-action candidates, record predicted local gain, true local gain, actual global gain, and realized KL.
- Analyze learned predicted-minus-true local gain and the frequency of negative global gains (classify gains below `-1e-12` as negative).
- Compare learned and oracle candidates within `(training_method, seed, step, state, action, method)` pairs. Join to checkpoint records using originating training method, seed, and step; verify the repeated checkpoint return and critic quality match.
- Summarize within each seed before reporting across-seed mean, sample SD, and individual seed values. Checkpoints and paired candidates are not independent replicates.
- Inspect relationships between critic quality and prediction errors descriptively. Do not interpret associations as causal effects.

The exact single-state performance-difference identity and learned/oracle separation are tested implementation invariants. They are not empirical hypotheses to be established from plots.

## Reproduction

Run from the repository root with uv installed:

```sh
uv sync --locked --python 3.11 --extra dev --extra analysis
uv run --locked ruff check .
uv run --locked pytest -q
```

The following commands are the **planned full Milestone 1 run**, not part of the Milestone 0 smoke check. Use a clean working tree or preserve its patch alongside the recorded revision. Keep `uv.lock` at that revision and record the Python/uv versions with retained results.

```sh
git rev-parse HEAD > experiments/001-critic-reliability/data/source_revision.txt
uv run --locked policy-lab --methods qac_full --steps 5000 --eval-every 100 --actor-lr 0.20 --critic-lr 0.12 --seeds 0 1 2 3 4 5 6 7 8 9 --out-dir experiments/001-critic-reliability/data
uv run --locked policy-lab-plot experiments/001-critic-reliability/data --out-dir experiments/001-critic-reliability/analysis
```

`data/` receives `config.json`, `checkpoints.csv`, and `counterfactual_updates.csv`; the revision file identifies the source. The stored KL limit is unused by the primary full-action operator. `analysis/` receives four PNGs, including `learned_vs_oracle_gain.png` for the primary paired comparison. The two other candidate scatter plots include supplementary operators and must not be presented as experiment 001's primary analysis. Additional seed-level tables and written analysis are to be produced during Milestone 1 using these CSVs and the rules above. Do not mix outputs from different configurations in one directory; the commands overwrite matching filenames.

For a short smoke run, use the [main README commands](../../README.md#setup-and-verification), which write under ignored `artifacts/smoke/` and do not populate research results.

## Results

Pending. No generated research measurements or analyses are committed. A successful smoke run establishes execution and file generation only.

## Interpretation

Pending measurements. An oracle-guided finite step is a reference comparison, not a guaranteed upper bound on realized return improvement. Zero or negative findings should be reported with the same settings and provenance as positive findings.

## Limitations

One handcrafted tabular MDP, one learner, and fixed rates limit generality. Visited checkpoint states and repeated measurements are dependent. Critic quality is uniformly weighted, whereas training is visitation weighted. Learned and exact guidance can yield different update sizes at the same actor rate; this experiment does not isolate direction from step size. Exact expected discounted return does not measure empirical return variance. Neural approximation, matched-KL comparisons, critic corruption, and causal claims are outside the initial scope.
