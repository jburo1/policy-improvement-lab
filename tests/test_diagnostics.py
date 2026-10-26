import numpy as np

from policy_improvement_lab.diagnostics import counterfactual_updates, critic_quality
from policy_improvement_lab.mdp import make_branching_mdp
from policy_improvement_lab.operators import softmax


def test_exact_critic_has_perfect_ranking_and_zero_regret():
    mdp = make_branching_mdp()
    _, q = mdp.exact_values(softmax(np.zeros((mdp.n_states, mdp.n_actions))))
    summary = critic_quality(mdp, q, q)
    assert summary.rmse == 0
    assert summary.ranking_accuracy == 1
    assert summary.consequential_ranking_regret == 0


def test_counterfactual_same_policy_and_critic_and_single_state_invariant():
    mdp = make_branching_mdp()
    logits = np.zeros((mdp.n_states, mdp.n_actions))
    _, exact_q = mdp.exact_values(softmax(logits))
    old_logits, old_q = logits.copy(), exact_q.copy()
    cases = counterfactual_updates(
        mdp, logits, exact_q, 0, 0, 0.5, 0.001, 0, 100, training_method="qac_full",
    )
    np.testing.assert_array_equal(logits, old_logits)
    np.testing.assert_array_equal(exact_q, old_q)
    assert len(cases) == 6
    for case in cases:
        assert case["training_method"] == "qac_full"
        assert case["step"] == 100
        assert case["checkpoint_expected_return"] == mdp.performance(softmax(logits))
        # A single-state update cannot lower J if it improves that state's true
        # local action value (performance-difference lemma).
        local = case["true_local_gain"]
        actual = case["actual_performance_gain"]
        assert local * actual >= -1e-12
        if case["method"] == "qac_full_kl":
            assert case["max_state_kl"] <= 0.001 + 1e-12
