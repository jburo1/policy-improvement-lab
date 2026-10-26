import numpy as np

from policy_improvement_lab.mdp import make_branching_mdp


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
