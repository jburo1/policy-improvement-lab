"""Exact critic and counterfactual single-update diagnostics."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .mdp import FiniteMDP
from .operators import QAC_METHODS, kl_new_old, softmax, update_logits


@dataclass(frozen=True)
class CriticQuality:
    rmse: float
    ranking_accuracy: float
    consequential_ranking_regret: float


def critic_quality(
    mdp: FiniteMDP, predicted: NDArray[np.float64], true: NDArray[np.float64],
    gap_tolerance: float = 1e-6,
) -> CriticQuality:
    """Uniform over decision states/actions; excludes near-tied *true* rankings."""
    active = mdp.active_states
    error = predicted[active] - true[active]
    rmse = float(np.sqrt(np.mean(error**2)))
    pair_total, pair_correct = 0, 0
    for s in active:
        for a in range(mdp.n_actions):
            for b in range(a + 1, mdp.n_actions):
                diff = true[s, a] - true[s, b]
                if abs(diff) <= gap_tolerance:
                    continue
                pair_total += 1
                pair_correct += (predicted[s, a] - predicted[s, b]) * diff > 0
    accuracy = float(pair_correct / pair_total) if pair_total else float("nan")
    preferred = np.argmax(predicted[active], axis=1)
    regret = float(np.mean(np.max(true[active], axis=1) - true[active, preferred]))
    return CriticQuality(rmse, accuracy, regret)


def counterfactual_updates(
    mdp: FiniteMDP,
    logits: NDArray[np.float64],
    critic_q: NDArray[np.float64],
    state: int,
    action: int,
    actor_lr: float,
    kl_limit: float,
    seed: int,
    step: int,
) -> list[dict[str, str | float | int]]:
    """Compare candidate actor updates from the *same* policy and critic.

    Critic sources: learned and exact/oracle. Oracle access is exclusively
    diagnostic. The learned QAC training never observes exact Q^pi.
    """
    old_pi = softmax(logits)
    old_j = mdp.performance(old_pi)
    _, q_true = mdp.exact_values(old_pi)
    quality = critic_quality(mdp, critic_q, q_true)
    records: list[dict[str, str | float | int]] = []
    for source, guidance in (("learned", critic_q), ("oracle", q_true)):
        for method in QAC_METHODS:
            new_logits = update_logits(logits, state, method, action, guidance, actor_lr, kl_limit)
            new_pi = softmax(new_logits)
            delta_probs = new_pi[state] - old_pi[state]
            predicted_gain = float(delta_probs @ guidance[state])
            true_local_gain = float(delta_probs @ q_true[state])
            actual_gain = float(mdp.performance(new_pi) - old_j)
            kls = [kl_new_old(new_pi[s], old_pi[s]) for s in mdp.active_states]
            records.append({
                "seed": seed,
                "step": step,
                "state": state,
                "action": action,
                "method": method,
                "critic_source": source,
                "predicted_local_gain": predicted_gain,
                "true_local_gain": true_local_gain,
                "actual_performance_gain": actual_gain,
                "state_kl": kl_new_old(new_pi[state], old_pi[state]),
                "max_state_kl": float(max(kls)),
                **asdict(quality),
            })
    return records
