"""An exactly evaluable finite discounted MDP.

All action-value and policy-performance references in this module are exact
under the specified MDP model. These exact values are *diagnostics only*:
learners do not get access to them during training.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FiniteMDP:
    transition: NDArray[np.float64]  # [S, A, S']
    reward: NDArray[np.float64]  # [S, A, S'], conditional on the next state
    initial_distribution: NDArray[np.float64]  # [S]
    gamma: float = 0.95
    terminal_state: int = 5

    def __post_init__(self) -> None:
        s, a, s_next = self.transition.shape
        if s != s_next or a < 2 or self.reward.shape != self.transition.shape:
            raise ValueError("Invalid transition/reward dimensions")
        if self.initial_distribution.shape != (s,):
            raise ValueError("Invalid initial distribution dimensions")
        if not np.allclose(self.transition.sum(axis=2), 1.0):
            raise ValueError("Transition probabilities must sum to one")
        if (np.any(self.transition < 0) or np.any(self.initial_distribution < 0)
                or not np.isclose(self.initial_distribution.sum(), 1)):
            raise ValueError("Invalid probability values")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0,1)")
        if self.terminal_state >= s or self.terminal_state < 0:
            raise ValueError("Invalid terminal state")
        terminal = np.zeros(s)
        terminal[self.terminal_state] = 1
        if not np.array_equal(self.transition[self.terminal_state], np.tile(terminal, (a, 1))):
            raise ValueError("Terminal state must be absorbing")
        if not np.isfinite(self.reward).all() or np.any(self.reward[self.terminal_state] != 0):
            raise ValueError("Rewards must be finite and zero after termination")

    @property
    def n_states(self) -> int:
        return self.transition.shape[0]

    @property
    def n_actions(self) -> int:
        return self.transition.shape[1]

    @property
    def active_states(self) -> NDArray[np.int64]:
        return np.array([s for s in range(self.n_states) if s != self.terminal_state])

    def step(self, state: int, action: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        next_state = int(rng.choice(self.n_states, p=self.transition[state, action]))
        reward = float(self.reward[state, action, next_state])
        return next_state, reward, next_state == self.terminal_state

    def exact_values(self, policy: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve V^pi = r_pi + gamma P_pi V^pi, then derive Q^pi."""
        if policy.shape != (self.n_states, self.n_actions):
            raise ValueError("Invalid policy shape")
        if not np.allclose(policy.sum(axis=1), 1) or np.any(policy < 0):
            raise ValueError("Invalid policy probabilities")
        expected_rewards = (self.transition * self.reward).sum(axis=2)
        p_pi = np.einsum("sa,san->sn", policy, self.transition)
        r_pi = np.einsum("sa,sa->s", policy, expected_rewards)
        value = np.linalg.solve(np.eye(self.n_states) - self.gamma * p_pi, r_pi)
        q_value = expected_rewards + self.gamma * np.einsum("san,n->sa", self.transition, value)
        return value, q_value

    def performance(self, policy: NDArray[np.float64]) -> float:
        value, _ = self.exact_values(policy)
        return float(self.initial_distribution @ value)

    def optimal_performance(self) -> float:
        """Diagnostic J*: policy iteration with exact Bellman policy evaluation."""
        policy = np.full((self.n_states, self.n_actions), 1 / self.n_actions)
        while True:
            value, q_value = self.exact_values(policy)
            # Only the optimality check has a numerical tolerance; the returned
            # value comes from a linear solve, not truncated value iteration.
            if np.max(q_value.max(axis=1) - value) <= 1e-12:
                return float(self.initial_distribution @ value)
            policy = np.eye(self.n_actions)[np.argmax(q_value, axis=1)]


def make_branching_mdp() -> FiniteMDP:
    """Five decision states, three actions, stochastic routes, delayed payoff.

    This transparent handcrafted benchmark is intended for debugging and
    mechanistic analysis, not for claiming broad algorithm performance.
    """
    s, a = 6, 3
    transition = np.zeros((s, a, s))
    reward = np.zeros((s, a, s))

    def branch(state: int, action: int, outcomes: dict[int, float], step_cost: float) -> None:
        for nxt, prob in outcomes.items():
            transition[state, action, nxt] = prob
            reward[state, action, nxt] = step_cost

    branch(0, 0, {1: 0.85, 2: 0.15}, -0.08)
    branch(0, 1, {2: 0.70, 1: 0.30}, -0.02)
    branch(0, 2, {0: 0.90, 1: 0.10}, -0.15)
    branch(1, 0, {4: 0.70, 3: 0.30}, 0.05)
    branch(1, 1, {3: 0.70, 2: 0.30}, 0.0)
    branch(1, 2, {0: 0.70, 1: 0.30}, -0.20)
    branch(2, 0, {3: 0.65, 4: 0.35}, 0.05)
    branch(2, 1, {4: 0.25, 2: 0.75}, -0.04)
    branch(2, 2, {0: 0.75, 3: 0.25}, -0.10)
    branch(3, 0, {4: 0.75, 0: 0.25}, -0.03)
    branch(3, 1, {4: 0.35, 2: 0.65}, -0.07)
    branch(3, 2, {0: 0.80, 3: 0.20}, -0.15)
    branch(4, 0, {5: 0.90, 3: 0.10}, -0.03)
    branch(4, 1, {5: 0.35, 2: 0.65}, -0.05)
    branch(4, 2, {0: 0.90, 4: 0.10}, -0.20)
    reward[4, 0, 5] = 2.0
    reward[4, 1, 5] = 2.0
    for action in range(a):
        transition[5, action, 5] = 1.0
    initial = np.zeros(s)
    initial[0] = 1.0
    return FiniteMDP(transition, reward, initial)
