import numpy as np
import pytest

from policy_improvement_lab.diagnostics import (
    GRADIENT_ZERO_TOLERANCE,
    counterfactual_updates,
    critic_quality,
    gradient_alignment,
)
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
        assert case["gradient_cosine_similarity"] == pytest.approx(1)
        assert case["learned_gradient_norm"] == case["oracle_gradient_norm"]
        # A single-state update cannot lower J if it improves that state's true
        # local action value (performance-difference lemma).
        local = case["true_local_gain"]
        actual = case["actual_performance_gain"]
        assert local * actual >= -1e-12
        if case["method"] == "qac_full_kl":
            assert case["max_state_kl"] <= 0.001 + 1e-12


@pytest.mark.parametrize("exact, expected", [([1, -1, 0], 1), ([-1, 1, 0], -1), ([1, 1, -2], 0)])
def test_gradient_alignment_directions(exact, expected):
    p = np.full(3, 1 / 3)
    learned = np.array([1., -1., 0.])
    result = gradient_alignment(p, learned, np.array(exact))
    assert result.gradient_cosine_similarity == pytest.approx(expected, abs=1e-15)
    assert result.learned_gradient_norm == pytest.approx(np.sqrt(2) / 3)
    # State-constant offsets cannot change a policy gradient.
    shifted = gradient_alignment(p, learned + 5, np.array(exact) - 7)
    assert shifted.gradient_cosine_similarity == pytest.approx(expected, abs=1e-15)


def test_gradient_alignment_matches_finite_difference_local_objective():
    logits = np.array([0.8, -0.4, 0.1])
    learned, exact = np.array([1.1, -0.2, 0.7]), np.array([-0.4, 0.9, 0.2])
    gradients = []
    for q in (learned, exact):
        gradient = []
        for direction in np.eye(3) * 1e-5:
            gradient.append((softmax(logits + direction) @ q - softmax(logits - direction) @ q) / 2e-5)
        gradients.append(np.array(gradient))
    result = gradient_alignment(softmax(logits), learned, exact)
    norms = [np.linalg.norm(g) for g in gradients]
    assert result.learned_gradient_norm == pytest.approx(norms[0])
    assert result.oracle_gradient_norm == pytest.approx(norms[1])
    assert result.gradient_cosine_similarity == pytest.approx(gradients[0] @ gradients[1] / np.prod(norms))


@pytest.mark.parametrize("learned, exact", [
    ([0, 0, 0], [1, -1, 0]), ([1, -1, 0], [2, 2, 2]), ([0, 0, 0], [0, 0, 0]),
    ([1e-14, -1e-14, 0], [1, -1, 0]),
])
def test_zero_gradient_alignment_is_explicitly_undefined(learned, exact):
    result = gradient_alignment(np.full(3, 1 / 3), np.array(learned), np.array(exact))
    assert min(result.learned_gradient_norm, result.oracle_gradient_norm) <= GRADIENT_ZERO_TOLERANCE
    assert np.isnan(result.gradient_cosine_similarity)


def test_initial_zero_critic_records_norms_and_undefined_cosine():
    mdp = make_branching_mdp()
    zeros = np.zeros((mdp.n_states, mdp.n_actions))
    rows = counterfactual_updates(mdp, zeros, zeros, 0, 0, .2, .002, 0, 0, training_method="qac_full")
    for row in rows:
        assert row["learned_gradient_norm"] == 0
        assert row["oracle_gradient_norm"] > 0
        assert np.isnan(row["gradient_cosine_similarity"])
