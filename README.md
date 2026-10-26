# Policy Improvement Lab

During graduate school, I studied actor-critic deep reinforcement learning architectures and investigated whether an auxiliry loss term which directly compares policy and value funtion output would allow such algorithms to achieve comparable performance on benchmark RL tasks with less experiential data. The results showed that algorithms which incorporate a concordance term which evaluates the relative ordering of policy and value predictions can improve sample efficiency in certain configurations, however, the method was not fully generalizable and in certain cases caused training to fail. This reboot takes a more principled approach towards policy improvement analysis in ac architectures, and investigates a more fundamental question:

**Research question:** How should an actor use imperfect critic-derived action values to make efficient, stable, and appropriately conservative policy improvements?

The first milestone is a **tabular, exactly evaluable** MDP, not a new RL algorithm or a general-purpose framework. It compares online one-step actor–critic (A2C-style), sampled-action QAC, full-action QAC, and a KL-backtracked full-action QAC baseline.

## What is implemented

- A stochastic finite discounted MDP with **five decision states, three actions, and an absorbing terminal state**.
- Exact policy evaluation with the Bellman equations, providing reference \(V^\pi\), \(Q^\pi\), and performance \(J(\pi)\).
- One-step online actor–critic (`ac_td0`) with a tabular state-value critic.
- QAC with a tabular action-value critic trained with on-policy one-step **Expected SARSA**.
- Three QAC actor operators: sampled-action gradient, exact action expectation, and full-action gradient with per-state KL backtracking.
- Same-checkpoint counterfactual actor updates using a **frozen learned critic** or an **oracle/exact critic for diagnosis only**.
- Critic RMSE, action-ranking accuracy, action-selection regret, predicted local gain, true local gain, actual return improvement, and realized KL.
- Seeded experiments, CSV outputs, tests, and GitHub Actions CI.

## Research goals, not claims

1. Characterize whether an imperfect learned critic mispredicts the effects of actor updates.
2. Compare the variance and reliability of sampled-action and full-action policy improvements.
3. Investigate whether restricting actor-update size reduces negative effects of critic approximation errors, and at what efficiency cost.
4. Test controlled critic corruption before designing new methods for estimating critic reliability.

## Prior work

- Sutton et al., *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (2000).
- Ciosek & Whiteson, *Expected Policy Gradients* (JMLR 2020).
- Schulman et al., *Trust Region Policy Optimization* (ICML 2015).
- Abdolmaleki et al., *Maximum a Posteriori Policy Optimisation* (2018).
- Fujimoto et al., *Why Should I Trust You, Bellman?* (ICML 2022).
