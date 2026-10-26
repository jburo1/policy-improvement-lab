"""Transparent tabular policy improvement operators.

Actor policy logits are tabular. The sampled and full-action operators are
ordinary policy-gradient ascent steps with the *same* nominal actor LR.
KL-backtracked full-action QAC uses the same direction but scales the step
until a per-state forward KL cap is respected; it is not TRPO or MPO.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Method = Literal["ac_td0", "qac_sampled", "qac_full", "qac_full_kl"]
QAC_METHODS: tuple[Method, ...] = ("qac_sampled", "qac_full", "qac_full_kl")
ALL_METHODS: tuple[Method, ...] = ("ac_td0", *QAC_METHODS)


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(shift)
    return weights / weights.sum(axis=-1, keepdims=True)


def kl_new_old(new: NDArray[np.float64], old: NDArray[np.float64]) -> float:
    """D_KL(pi_new || pi_old), for strictly positive softmax policies."""
    return float(np.sum(new * (np.log(new) - np.log(old))))


def update_logits(
    logits: NDArray[np.float64],
    state: int,
    method: Method,
    action: int,
    critic_q: NDArray[np.float64],
    actor_lr: float,
    kl_limit: float,
    td_advantage: float | None = None,
) -> NDArray[np.float64]:
    """Return a new logits table; updates only the visited state's row."""
    updated = logits.copy()
    p = softmax(logits[state])
    if method == "ac_td0":
        if td_advantage is None:
            raise ValueError("One-step actor-critic requires its bootstrapped advantage")
        gradient = td_advantage * (np.eye(len(p))[action] - p)
    elif method == "qac_sampled":
        v_hat = float(p @ critic_q[state])
        gradient = (critic_q[state, action] - v_hat) * (np.eye(len(p))[action] - p)
    elif method in ("qac_full", "qac_full_kl"):
        gradient = p * (critic_q[state] - float(p @ critic_q[state]))
    else:
        raise ValueError(f"Unknown method: {method}")

    step = actor_lr * gradient
    if method == "qac_full_kl":
        if kl_limit <= 0:
            raise ValueError("kl_limit must be positive")
        for _ in range(64):
            candidate = softmax(logits[state] + step)
            if kl_new_old(candidate, p) <= kl_limit + 1e-12:
                break
            step *= 0.5
        else:
            raise ArithmeticError("Unable to satisfy the KL bound")
    updated[state] += step
    return updated
