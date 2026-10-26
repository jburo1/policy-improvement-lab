import numpy as np

from policy_improvement_lab.operators import kl_new_old, softmax, update_logits


def test_softmax_normalizes_and_is_finite():
    p = softmax(np.array([[1001.0, 1002.0, 1003.0], [-1001.0, -1000.0, -1002.0]]))
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1)


def test_full_action_gradient_moves_toward_higher_q():
    logits = np.zeros((2, 3))
    q = np.array([[2.0, 1.0, -3.0], [0.0, 0.0, 0.0]])
    result = update_logits(logits, 0, "qac_full", 0, q, actor_lr=1.0, kl_limit=0.1)
    assert softmax(result)[0, 0] > 1 / 3
    assert np.array_equal(result[1], logits[1])


def test_backtracking_respects_kl_limit():
    logits = np.zeros((2, 3))
    q = np.array([[100.0, 0.0, -100.0], [0.0, 0.0, 0.0]])
    limit = 0.0001
    changed = update_logits(logits, 0, "qac_full_kl", 0, q, actor_lr=10.0, kl_limit=limit)
    assert 0 < kl_new_old(softmax(changed[0]), softmax(logits[0])) <= limit + 1e-12


def test_ac_td0_sampled_gradient_with_positive_advantage():
    logits = np.zeros((2, 3))
    changed = update_logits(logits, 0, "ac_td0", 1, np.zeros_like(logits), 0.5, 0.1, 2.0)
    assert softmax(changed)[0, 1] > 1 / 3
