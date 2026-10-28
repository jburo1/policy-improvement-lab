"""Scientific controls, deterministic draws, isolation, and exact EXP002 metrics."""

import json

import numpy as np
import pytest

from policy_improvement_lab.controlled_error import (
    ADDITIVE_BIAS,
    GAP_FLOOR,
    NOISE_MULTIPLIERS,
    NOISE_REPLICATES,
    checkpoint_candidates,
    construct_guidance,
    evaluate_candidate,
    gap_scales,
    noise_seed,
    run_experiment,
)
from policy_improvement_lab.diagnostics import critic_quality
from policy_improvement_lab.mdp import make_branching_mdp
from policy_improvement_lab.operators import softmax, update_logits
from policy_improvement_lab.training import RunConfig, train


def measure_candidate(mdp, logits, exact_q, state, guidance, cfg):
    old_pi = softmax(logits)
    return evaluate_candidate(mdp, logits, exact_q, state, guidance, cfg,
                              old_pi=old_pi, old_j=mdp.performance(old_pi))


@pytest.fixture
def frozen():
    mdp = make_branching_mdp()
    logits = np.random.default_rng(5).normal(size=(mdp.n_states, mdp.n_actions))
    _, exact = mdp.exact_values(softmax(logits))
    return mdp, logits, exact, RunConfig(steps=10, eval_every=10)


def test_gap_scale_and_near_ties():
    q = np.array([[0., 1., 3.], [2., 2., 2.], [1., 1. + 1e-9, 1. - 1e-9]])
    np.testing.assert_allclose(gap_scales(q), [np.sqrt(14 / 3), GAP_FLOOR, GAP_FLOOR])
    np.testing.assert_allclose(gap_scales(q + 100), gap_scales(q))


def test_guidance_construction_and_independent_deterministic_draws(frozen):
    mdp, _, exact, _ = frozen
    original = exact.copy()
    active = mdp.active_states
    keys = [noise_seed(7, 100, m, r) for m in NOISE_MULTIPLIERS for r in range(NOISE_REPLICATES)]
    assert len(set(keys)) == len(keys)
    assert noise_seed(7, 100, 1, 0) != noise_seed(8, 100, 1, 0)
    assert noise_seed(7, 100, 1, 0) != noise_seed(7, 200, 1, 0)
    assert noise_seed(7, 100, 1, 0) != noise_seed(7, 100, 1, 0, root=1)
    for magnitude in reversed(NOISE_MULTIPLIERS):
        sample_seed = noise_seed(7, 100, magnitude, 2)
        first = construct_guidance(mdp, exact, "gaussian_noise", magnitude, sample_seed=sample_seed)
        second = construct_guidance(mdp, exact, "gaussian_noise", magnitude, sample_seed=sample_seed)
        np.testing.assert_array_equal(first, second)
        z = np.random.default_rng(sample_seed).standard_normal((len(active), mdp.n_actions))
        np.testing.assert_allclose(first[active],
                                   exact[active] + magnitude * gap_scales(exact)[active, None] * z)
        np.testing.assert_array_equal(first[mdp.terminal_state], exact[mdp.terminal_state])
    np.testing.assert_array_equal(construct_guidance(mdp, exact, "oracle"), exact)
    np.testing.assert_allclose(construct_guidance(mdp, exact, "additive_bias", 10)[active],
                               exact[active] + 10)
    np.testing.assert_array_equal(construct_guidance(mdp, exact, "reversed_guidance")[active], -exact[active])
    np.testing.assert_array_equal(exact, original)
    with pytest.raises(ValueError, match="explicit sample seed"):
        construct_guidance(mdp, exact, "gaussian_noise")
    with pytest.raises(ValueError, match="Unknown condition"):
        construct_guidance(mdp, exact, "other")


@pytest.mark.parametrize("state", range(5))
def test_bias_invariance_and_reversed_gradient(frozen, state):
    mdp, logits, exact, cfg = frozen
    oracle = update_logits(logits, state, "qac_full", 0, exact, cfg.actor_lr, cfg.kl_limit)
    for bias in (0., 1., ADDITIVE_BIAS):
        guidance = construct_guidance(mdp, exact, "additive_bias", bias)
        candidate = update_logits(logits, state, "qac_full", 0, guidance, cfg.actor_lr, cfg.kl_limit)
        np.testing.assert_allclose(candidate, oracle, rtol=0, atol=1e-15)
        metrics = measure_candidate(mdp, logits, exact, state, guidance, cfg)
        assert metrics["local_rmse"] == pytest.approx(abs(bias), abs=1e-15)
        assert metrics["local_ranking_accuracy"] == 1
        assert metrics["gradient_cosine_similarity"] == pytest.approx(1)
    reverse = construct_guidance(mdp, exact, "reversed_guidance")
    candidate = update_logits(logits, state, "qac_full", 0, reverse, cfg.actor_lr, cfg.kl_limit)
    np.testing.assert_allclose(candidate - logits, -(oracle - logits), rtol=0, atol=2e-16)
    metrics = measure_candidate(mdp, logits, exact, state, reverse, cfg)
    assert metrics["gradient_cosine_similarity"] == pytest.approx(-1)
    assert metrics["local_ranking_accuracy"] == 0


def test_metrics_match_exact_local_and_global_definitions(frozen):
    mdp, logits, exact, cfg = frozen
    state = 2
    guidance = construct_guidance(mdp, exact, "gaussian_noise", 2, sample_seed=42)
    old_pi = softmax(logits)
    p = old_pi[state]
    gradient = p * (guidance[state] - p @ guidance[state])
    new_pi = old_pi.copy()
    new_pi[state] = softmax(logits[state] + cfg.actor_lr * gradient)
    metrics = measure_candidate(mdp, logits, exact, state, guidance, cfg)
    local_gain = (new_pi[state] - p) @ exact[state]
    assert metrics["true_local_gain"] == pytest.approx(local_gain)
    assert metrics["actual_performance_gain"] == pytest.approx(
        mdp.performance(new_pi) - mdp.performance(old_pi))
    assert metrics["local_rmse"] == pytest.approx(np.sqrt(np.mean((guidance[state] - exact[state]) ** 2)))
    a, b = np.triu_indices(mdp.n_actions, k=1)
    true_gaps = exact[state, a] - exact[state, b]
    eligible = abs(true_gaps) > 1e-6
    assert metrics["local_ranking_accuracy"] == pytest.approx(
        np.mean(((guidance[state, a] - guidance[state, b]) * true_gaps > 0)[eligible]))
    assert metrics["state_kl"] == pytest.approx(np.sum(new_pi[state] * np.log(new_pi[state] / p)))
    transition = np.einsum("sa,san->sn", new_pi, mdp.transition)
    occupancy = np.linalg.solve((np.eye(mdp.n_states) - mdp.gamma * transition).T, mdp.initial_distribution)
    assert metrics["actual_performance_gain"] == pytest.approx(occupancy[state] * local_gain, abs=1e-14)


def test_tied_values_have_undefined_rank_and_alignment_and_zero_update(frozen):
    mdp, logits, _, cfg = frozen
    exact = np.zeros_like(logits)
    metrics = measure_candidate(mdp, logits, exact, 0, exact, cfg)
    assert np.isnan(metrics["local_ranking_accuracy"])
    assert np.isnan(metrics["gradient_cosine_similarity"])
    assert metrics["guidance_gradient_norm"] == metrics["oracle_gradient_norm"] == 0
    assert metrics["actual_performance_gain"] == metrics["true_local_gain"] == metrics["state_kl"] == 0
    assert metrics["effectively_zero"] == 1
    noisy = construct_guidance(mdp, exact, "gaussian_noise", 1, sample_seed=42)
    assert np.any(noisy[mdp.active_states] != 0)  # The documented floor preserves nonzero noise.


def test_guidance_at_other_states_cannot_affect_candidate_metrics(frozen):
    mdp, logits, exact, cfg = frozen
    guidance = construct_guidance(mdp, exact, "gaussian_noise", 2, sample_seed=42)
    expected = measure_candidate(mdp, logits, exact, 2, guidance, cfg)
    guidance[[0, 1, 3, 4]] += 1000
    assert measure_candidate(mdp, logits, exact, 2, guidance, cfg) == expected


def test_checkpoint_candidates_are_isolated_and_paired(frozen):
    mdp, logits, _, cfg = frozen
    original = logits.copy()
    first = checkpoint_candidates(mdp, logits, 0, 7, 100, cfg)
    second = checkpoint_candidates(mdp, logits, 0, 7, 100, cfg)
    np.testing.assert_array_equal(logits, original)
    assert first == second
    assert len(first) == 3 + len(NOISE_MULTIPLIERS) * NOISE_REPLICATES
    oracle = next(r for r in first if r["condition"] == "oracle")
    assert len([r for r in first if r["magnitude"] == 0]) == 1
    assert {r["magnitude"] for r in first if r["condition"] == "additive_bias"} == {10.}
    for row in first:
        assert (row["seed"], row["step"], row["state"]) == (7, 100, 0)
        assert row["oracle_minus_perturbed_gain"] == pytest.approx(
            oracle["actual_performance_gain"] - row["actual_performance_gain"])
        if row["condition"] == "additive_bias" or row["magnitude"] == 0:
            assert row["actual_performance_gain"] == pytest.approx(
                oracle["actual_performance_gain"], abs=1e-14)


def test_checkpoint_observer_cannot_change_exp001_trajectory(frozen, monkeypatch):
    mdp, _, _, cfg = frozen
    baseline, baseline_candidates = train(mdp, "qac_full", 7, cfg)
    seen = []

    def unused(*args, **kwargs):
        raise AssertionError("EXP002 must skip EXP001 counterfactuals")

    monkeypatch.setattr("policy_improvement_lab.training.counterfactual_updates", unused)

    def observe(step, logits, critic, state):
        seen.append((step, logits.copy(), critic.copy(), state))
        checkpoint_candidates(mdp, logits, state, 7, step, cfg)
        logits[:] = 123
        critic[:] = -456

    measured, candidates = train(mdp, "qac_full", 7, cfg, checkpoint_callback=observe,
                                 measure_counterfactuals=False)
    assert measured == baseline
    assert candidates == []
    assert [s[0] for s in seen] == [0, 10]
    assert seen[0][3] == 0
    np.testing.assert_array_equal(seen[0][1], 0)
    np.testing.assert_array_equal(seen[0][2], 0)
    for step, logits, critic, state in seen:
        full = [r for r in baseline_candidates if r["step"] == step and r["method"] == "qac_full"]
        assert {r["state"] for r in full} == {state}
        _, exact = mdp.exact_values(softmax(logits))
        quality = critic_quality(mdp, critic, exact)
        assert quality.rmse == next(r["rmse"] for r in baseline if r["step"] == step)
        oracle = measure_candidate(mdp, logits, exact, state, exact, cfg)
        assert oracle["actual_performance_gain"] == next(
            r["actual_performance_gain"] for r in full if r["critic_source"] == "oracle")


def test_smoke_outputs_and_saved_checkpoint_reconstruction(tmp_path, frozen):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from policy_improvement_lab.controlled_error_plot import analyze, validate_pairs

    mdp, _, _, cfg = frozen
    data, analysis = tmp_path / "data", tmp_path / "analysis"
    run_experiment(data, cfg, [7])
    assert {p.name for p in data.iterdir()} == {
        "config.json", "source_revision.txt", "checkpoints.csv", "counterfactual_updates.csv",
        "frozen_checkpoints.npz",
    }
    config = json.loads((data / "config.json").read_text())
    checkpoints = pd.read_csv(data / "checkpoints.csv")
    updates = pd.read_csv(data / "counterfactual_updates.csv")
    assert set(updates.condition) == {"oracle", "additive_bias", "gaussian_noise", "reversed_guidance"}
    assert len(checkpoints) == 2
    assert len(updates) == 246
    with np.load(data / "frozen_checkpoints.npz") as saved:
        assert saved["run_id"].item() == config["run_id"]
        for i, step in enumerate(saved["step"]):
            reconstructed = checkpoint_candidates(mdp, saved["logits"][i], int(saved["state"][i]),
                                                 int(saved["seed"][i]), int(step), cfg)
            actual = updates.loc[updates.step.eq(step)]
            for metric in ("actual_performance_gain", "local_rmse", "oracle_minus_perturbed_gain"):
                np.testing.assert_allclose(actual[metric], [r[metric] for r in reconstructed], atol=1e-15)
    analyze(data, analysis)
    assert {p.name for p in analysis.iterdir()} == {
        "checkpoint_summary.csv", "seed_summary.csv", "summary.csv", "noise_response.png",
        "gradient_alignment_vs_improvement.png",
    }
    assert pd.read_csv(analysis / "summary.csv").harmful_frequency_std.isna().all()
    for broken in (updates.iloc[1:], pd.concat([updates, updates.iloc[:1]])):
        with pytest.raises(ValueError, match="checkpoint candidates/replicates"):
            validate_pairs(checkpoints, broken, config)
    broken = updates.copy()
    broken.loc[0, "oracle_performance_gain"] += 1
    with pytest.raises(ValueError, match="same frozen checkpoint"):
        validate_pairs(checkpoints, broken, config)
    run_experiment(tmp_path / "repeat", cfg, [7])
    assert (data / "counterfactual_updates.csv").read_bytes() == (
        tmp_path / "repeat" / "counterfactual_updates.csv").read_bytes()


def test_hierarchical_statistics_do_not_pool_draws_or_checkpoints():
    pd = pytest.importorskip("pandas")
    from policy_improvement_lab.controlled_error_plot import (
        MEASUREMENTS,
        summarize_across_seeds,
        summarize_checkpoints,
        summarize_seeds,
    )

    # Deliberately unequal draw and checkpoint counts expose accidental pooling.
    rows = []
    for seed, step, gains in [(0, 0, [-1, -1, -1, 1]), (0, 100, [1]), (1, 0, [-1])]:
        for gain in gains:
            row = dict.fromkeys(MEASUREMENTS, 0.)
            row.update(run_id="test", seed=seed, step=step, state=0, condition="gaussian_noise", magnitude=1,
                       actual_performance_gain=gain, harmful=int(gain < 0), beneficial=int(gain > 0),
                       effectively_zero=0, guidance_gradient_norm=1, oracle_gradient_norm=1,
                       gradient_cosine_similarity=float("nan") if step == 0 else 1)
            rows.append(row)
    checkpoints = summarize_checkpoints(pd.DataFrame(rows))
    seeds = summarize_seeds(checkpoints).set_index("seed")
    np.testing.assert_allclose(seeds.harmful_frequency, [.375, 1])
    np.testing.assert_allclose(seeds.actual_performance_gain, [.25, -1])
    np.testing.assert_allclose(seeds.candidate_count, [5, 1])
    np.testing.assert_allclose(seeds.checkpoint_count, [2, 1])
    np.testing.assert_allclose(seeds.defined_alignment_frequency, [.5, 0])
    assert np.isnan(seeds.loc[1, "gradient_cosine_similarity"])
    summary = summarize_across_seeds(seeds.reset_index()).iloc[0]
    assert summary.harmful_frequency_mean == pytest.approx((.375 + 1) / 2)
    assert summary.harmful_frequency_std == pytest.approx(np.std([.375, 1], ddof=1))
    assert summary.gradient_cosine_similarity_count == 1
