"""Tabular online actor-critic and QAC on a small, exactly evaluable MDP."""

from dataclasses import dataclass

import numpy as np

from .diagnostics import counterfactual_updates, critic_quality
from .mdp import FiniteMDP
from .operators import Method, softmax, update_logits


@dataclass(frozen=True)
class RunConfig:
    steps: int = 5_000
    eval_every: int = 100
    actor_lr: float = 0.20
    critic_lr: float = 0.12
    kl_limit: float = 0.002

    def __post_init__(self) -> None:
        if min(self.steps, self.eval_every) <= 0:
            raise ValueError("Training steps and evaluation frequency must be positive")
        if min(self.actor_lr, self.critic_lr, self.kl_limit) <= 0:
            raise ValueError("Learning rates and KL limit must be positive")


def train(mdp: FiniteMDP, method: Method, seed: int, cfg: RunConfig) -> tuple[list[dict], list[dict]]:
    """Run one seeded learner; return checkpoint and counterfactual records.

    * Online actor-critic (A2C-style): one-step TD with a tabular V critic.
    * QAC: on-policy one-step Expected SARSA with a tabular Q critic.
    * Each actor update changes only the visited state's logit row.
    * Reference Q^pi is used for analysis, never for actual learning.
    """
    rng = np.random.default_rng(seed)
    logits = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
    q_hat = np.zeros_like(logits)
    v_hat = np.zeros(mdp.n_states, dtype=np.float64)
    state = int(rng.choice(mdp.n_states, p=mdp.initial_distribution))
    trajectory_metrics: list[dict] = []
    update_metrics: list[dict] = []

    def checkpoint(step: int) -> None:
        p = softmax(logits)
        record = {"seed": seed, "step": step, "method": method, "expected_return": mdp.performance(p)}
        if method != "ac_td0":
            _, true_q = mdp.exact_values(p)
            record.update(vars(critic_quality(mdp, q_hat, true_q)))
        trajectory_metrics.append(record)

    checkpoint(0)
    for step in range(1, cfg.steps + 1):
        p = softmax(logits)
        action = int(rng.choice(mdp.n_actions, p=p[state]))
        next_state, reward, done = mdp.step(state, action, rng)

        if method == "ac_td0":
            td_target = reward + (0 if done else mdp.gamma * v_hat[next_state])
            td_error = td_target - v_hat[state]
            v_hat[state] += cfg.critic_lr * td_error
            logits = update_logits(
                logits, state, "ac_td0", action, q_hat, cfg.actor_lr, cfg.kl_limit,
                td_advantage=td_error,
            )
        else:
            expected_next = 0 if done else float(p[next_state] @ q_hat[next_state])
            td_target = reward + mdp.gamma * expected_next
            td_error = td_target - q_hat[state, action]
            q_hat[state, action] += cfg.critic_lr * td_error
            if step % cfg.eval_every == 0:
                update_metrics.extend(
                    counterfactual_updates(
                        mdp, logits, q_hat, state, action, cfg.actor_lr, cfg.kl_limit, seed, step
                    )
                )
            logits = update_logits(logits, state, method, action, q_hat, cfg.actor_lr, cfg.kl_limit)

        state = int(rng.choice(mdp.n_states, p=mdp.initial_distribution)) if done else next_state
        if step % cfg.eval_every == 0 or step == cfg.steps:
            checkpoint(step)
    return trajectory_metrics, update_metrics
