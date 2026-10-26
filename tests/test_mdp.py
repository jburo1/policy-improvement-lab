from itertools import product

import numpy as np
import pytest

from policy_improvement_lab.mdp import FiniteMDP, make_branching_mdp
from policy_improvement_lab.operators import softmax


def test_optimal_return_matches_exhaustive_deterministic_policy_evaluation():
    mdp = make_branching_mdp()
    returns = []
    for actions in product(range(mdp.n_actions), repeat=len(mdp.active_states)):
        policy = np.eye(mdp.n_actions)[list(actions) + [0]]
        returns.append(mdp.performance(policy))
    assert mdp.optimal_performance() == pytest.approx(max(returns), abs=1e-12)


@pytest.mark.parametrize("gamma", [0., .8, .99])
def test_optimal_return_closed_form_with_looping_and_terminating_actions(gamma):
    transition = np.array([[[.5, .5], [1., 0]], [[0, 1], [0, 1]]])
    reward = np.array([[[2., 2.], [.1, 0]], [[0, 0], [0, 0]]])
    mdp = FiniteMDP(transition, reward, np.array([.7, .3]), gamma=gamma, terminal_state=1)
    assert mdp.optimal_performance() == pytest.approx(.7 * max(2 / (1 - .5 * gamma), .1 / (1 - gamma)))


def test_exact_policy_evaluation_satisfies_bellman_equations():
    mdp = make_branching_mdp()
    pi = np.full((mdp.n_states, mdp.n_actions), 1 / mdp.n_actions)
    v, q = mdp.exact_values(pi)
    r_sa = (mdp.transition * mdp.reward).sum(axis=2)
    expected_q = r_sa + mdp.gamma * np.einsum("san,n->sa", mdp.transition, v)
    assert np.allclose(q, expected_q, atol=1e-10)
    assert np.allclose(v, (pi * q).sum(axis=1), atol=1e-10)
    assert np.isclose(v[mdp.terminal_state], 0)
    assert np.isfinite(mdp.performance(pi))


def test_transition_distribution_valid():
    mdp = make_branching_mdp()
    assert np.allclose(mdp.transition.sum(axis=-1), 1)
    assert mdp.active_states.size == mdp.n_states - 1


def test_exact_evaluation_matches_closed_form_with_conditional_rewards():
    # Half the transitions terminate with reward 3; the rest loop with reward -1.
    transition = np.array([[[0.5, 0.5], [0.5, 0.5]], [[0, 1], [0, 1]]])
    reward = np.array([[[-1, 3], [-1, 3]], [[0, 0], [0, 0]]])
    mdp = FiniteMDP(transition, reward, np.array([1.0, 0]), gamma=0.8, terminal_state=1)
    pi = np.full((2, 2), 0.5)
    v, q = mdp.exact_values(pi)
    np.testing.assert_allclose(v, [1 / (1 - 0.8 * 0.5), 0])
    np.testing.assert_allclose(q[0], v[0])
    assert mdp.performance(pi) == pytest.approx(v[0])


@pytest.mark.parametrize("seed", range(5))
def test_single_state_performance_difference_identity(seed):
    mdp = make_branching_mdp()
    rng = np.random.default_rng(seed)
    pi = softmax(rng.normal(size=(mdp.n_states, mdp.n_actions)))
    _, q = mdp.exact_values(pi)
    for state in mdp.active_states:
        candidate = pi.copy()
        candidate[state] = softmax(rng.normal(size=mdp.n_actions))
        p_new = np.einsum("sa,san->sn", candidate, mdp.transition)
        # Unnormalized discounted visitation under the NEW policy, including t=0.
        occupancy = np.linalg.solve(
            (np.eye(mdp.n_states) - mdp.gamma * p_new).T, mdp.initial_distribution,
        )
        local_gain = (candidate[state] - pi[state]) @ q[state]
        assert mdp.performance(candidate) - mdp.performance(pi) == pytest.approx(
            occupancy[state] * local_gain, abs=1e-12,
        )


@pytest.mark.parametrize("invalid_reward", [False, True])
def test_terminal_contract_matches_episodic_training(invalid_reward):
    mdp = make_branching_mdp()
    transition, reward = mdp.transition.copy(), mdp.reward.copy()
    if invalid_reward:
        reward[mdp.terminal_state, :, mdp.terminal_state] = 1
    else:
        transition[mdp.terminal_state] = transition[0]
    with pytest.raises(ValueError, match="Terminal state|termination"):
        FiniteMDP(transition, reward, mdp.initial_distribution)
