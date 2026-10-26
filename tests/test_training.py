import numpy as np
import pytest

import policy_improvement_lab.training as training
from policy_improvement_lab.diagnostics import critic_quality
from policy_improvement_lab.mdp import make_branching_mdp
from policy_improvement_lab.operators import ALL_METHODS, QAC_METHODS, softmax
from policy_improvement_lab.training import RunConfig, train


def test_smoke_and_reproducibility():
    mdp = make_branching_mdp()
    cfg = RunConfig(steps=40, eval_every=20)
    for method in ALL_METHODS:
        checkpoints, diagnostics = train(mdp, method, 7, cfg)
        assert len(checkpoints) == 3
        assert all(np.isfinite(r["expected_return"]) for r in checkpoints)
        if method == "ac_td0":
            assert not diagnostics
        else:
            assert len(diagnostics) == 18
            for case in diagnostics:
                checkpoint = next(r for r in checkpoints if r["step"] == case["step"])
                assert case["training_method"] == checkpoint["method"]
                assert case["checkpoint_expected_return"] == checkpoint["expected_return"]
                for metric in ("rmse", "ranking_accuracy", "consequential_ranking_regret"):
                    assert case[metric] == checkpoint[metric]
        checkpoints_2, diagnostics_2 = train(mdp, method, 7, cfg)
        assert checkpoints == checkpoints_2
        assert len(diagnostics) == len(diagnostics_2)
        for first, second in zip(diagnostics, diagnostics_2, strict=True):
            for key in first:
                if isinstance(first[key], str):
                    assert first[key] == second[key]
                else:
                    assert first[key] == pytest.approx(second[key], rel=0, abs=0, nan_ok=True)


@pytest.mark.parametrize("method", QAC_METHODS)
def test_actor_and_expected_sarsa_read_pre_transition_critic(method, monkeypatch):
    mdp = make_branching_mdp()
    cfg = RunConfig(steps=8, eval_every=3)
    # Independently replay the transition stream and track the pre-update critic.
    rng = np.random.default_rng(7)
    state = int(rng.choice(mdp.n_states, p=mdp.initial_distribution))
    expected_q = np.zeros((mdp.n_states, mdp.n_actions))
    actor = training.update_logits
    calls = 0
    final_policy = None

    def check(logits, actual_state, actual_method, action, q, actor_lr, kl_limit):
        nonlocal state, calls, final_policy
        p = softmax(logits)
        expected_action = int(rng.choice(mdp.n_actions, p=p[state]))
        assert (actual_state, action, actual_method) == (state, expected_action, method)
        np.testing.assert_allclose(q, expected_q, atol=1e-15)
        next_state, reward, done = mdp.step(state, action, rng)
        bootstrap = 0 if done else mdp.gamma * (p[next_state] @ expected_q[next_state])
        expected_q[state, action] += cfg.critic_lr * (reward + bootstrap - expected_q[state, action])
        state = int(rng.choice(mdp.n_states, p=mdp.initial_distribution)) if done else next_state
        calls += 1
        result = actor(logits, actual_state, actual_method, action, q, actor_lr, kl_limit)
        final_policy = softmax(result)
        return result

    monkeypatch.setattr(training, "update_logits", check)
    history, diagnostics = train(mdp, method, 7, cfg)
    assert calls == cfg.steps
    assert [row["step"] for row in history] == [0, 3, 6, 8]
    assert {row["step"] for row in diagnostics} == {0, 3, 6, 8}
    _, exact_q = mdp.exact_values(final_policy)
    quality = critic_quality(mdp, expected_q, exact_q)
    assert history[-1]["rmse"] == pytest.approx(quality.rmse)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_oracle_measurements_and_evaluation_cadence_cannot_change_training(method, monkeypatch):
    mdp = make_branching_mdp()
    actor = training.update_logits
    policies = []

    def capture(*args, **kwargs):
        result = actor(*args, **kwargs)
        policies.append(result.copy())
        return result

    monkeypatch.setattr(training, "update_logits", capture)
    reference, _ = train(mdp, method, 7, RunConfig(steps=41, eval_every=20))
    reference_policies = policies.copy()
    policies.clear()
    original = mdp.exact_values

    def corrupted_oracle(policy):
        v, q = original(policy)
        return v + 100, -q * 17

    monkeypatch.setattr(type(mdp), "exact_values", lambda self, pi: corrupted_oracle(pi))
    measured, _ = train(mdp, method, 7, RunConfig(steps=41, eval_every=7))
    np.testing.assert_array_equal(policies, reference_policies)
    assert measured[-1]["expected_return"] - 100 == pytest.approx(reference[-1]["expected_return"])
    # Critic quality is diagnostic and should reflect the corruption.
    if method != "ac_td0":
        assert measured[-1]["rmse"] != reference[-1]["rmse"]
