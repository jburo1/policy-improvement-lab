"""Protect Experiment 001 filtering, pairing, and seed-level statistics."""

import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("seaborn")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from policy_improvement_lab.cli import write_csv  # noqa: E402
from policy_improvement_lab.mdp import make_branching_mdp  # noqa: E402
from policy_improvement_lab.plot import (  # noqa: E402
    METHOD_LABELS,
    main,
    pair_candidates,
    update_rates,
)

FIGURES = {
    "learning_curves.png", "predicted_vs_true_gain.png", "update_kl_vs_improvement.png",
    "learned_vs_oracle_gain.png", "harmful_update_frequency.png", "critic_rmse_vs_prediction_error.png",
    "critic_quality_over_training.png", "gradient_alignment_over_training.png",
    "prediction_error_over_training.png",
}


def candidate(seed, step, source, gain, *, origin="qac_full", method="qac_full", error=1):
    return {
        "method": method, "training_method": origin, "seed": seed,
        "step": step, "state": 0, "action": 0, "critic_source": source,
        "predicted_local_gain": error, "true_local_gain": error * 2,
        "state_kl": error / 100, "actual_performance_gain": gain, "rmse": error * 3,
        "learned_gradient_norm": 0 if step == 0 else 1, "oracle_gradient_norm": 1,
        "gradient_cosine_similarity": float("nan") if step == 0 else .5,
    }


def checkpoint(seed, step, expected_return, method="qac_full"):
    return {"method": method, "seed": seed, "step": step, "expected_return": expected_return,
            "rmse": seed + step / 1000, "ranking_accuracy": .5 + seed / 10,
            "consequential_ranking_regret": .2 + seed / 10}


def test_plot_statistics_and_candidate_filtering(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write_csv(data / "checkpoints.csv", [
        checkpoint(1, 10, 6), checkpoint(0, 1000, 2), checkpoint(0, 0, 1), checkpoint(0, 10, 2),
        checkpoint(0, 0, 999, "ac_td0"), checkpoint(1, 10, 999, "qac_sampled"),
    ])
    learned_gains = [-2e-12, -1e-12, 2, 4]
    # Oracle rows are deliberately reversed: pairing must use keys, not row position.
    updates = [
        candidate(seed, step, "learned", gain, error=error)
        for seed, step, gain, error in zip(
            [0, 0, 0, 1], [0, 10, 1000, 10], learned_gains, [1, 2, 3, 8], strict=True,
        )
    ] + [
        candidate(seed, step, "oracle", gain, error=999)
        for seed, step, gain in [(1, 10, 40), (0, 1000, 30), (0, 10, 20), (0, 0, 10)]
    ]
    # Include both overestimation and underestimation without changing absolute errors.
    updates[1]["predicted_local_gain"] = 6
    # Same checkpoint identifiers, but outside the experimental scope.
    for source in ("learned", "oracle"):
        updates.extend([
            candidate(0, 0, source, 999, origin="qac_sampled", error=999),
            candidate(0, 0, source, 999, method="qac_sampled", error=999),
            candidate(0, 0, source, 999, method="qac_full_kl", error=999),
        ])
    write_csv(data / "counterfactual_updates.csv", updates)
    raw = {path: path.read_bytes() for path in data.iterdir()}
    figures = {}
    savefig = Figure.savefig

    def capture(fig, path, **kwargs):
        figures[path.name] = fig
        savefig(fig, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    output = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", ["policy-lab-plot", str(data), "--out-dir", str(output)])
    main()

    assert set(figures) == FIGURES
    assert {path.name for path in output.iterdir()} == FIGURES | {"summary.csv"}
    for name in figures:
        assert (output / name).stat().st_size > 0
    assert {path: path.read_bytes() for path in data.iterdir()} == raw

    curves = figures["learning_curves.png"].axes[0]
    lines = {line.get_label(): line for line in curves.lines}
    assert set(lines) == {METHOD_LABELS["qac_full"], "Optimal J*"}
    np.testing.assert_allclose(lines["Optimal J*"].get_ydata(), make_branching_mdp().optimal_performance())
    np.testing.assert_allclose(lines[METHOD_LABELS["qac_full"]].get_xydata(), [[0, 1], [10, 4], [1000, 2]])
    # Sample SD is sqrt(8); a singleton has a zero-width plotted band.
    qac_band = curves.collections[0].get_paths()[0].vertices
    at_step_10 = qac_band[qac_band[:, 0] == 10, 1]
    assert at_step_10.min() == pytest.approx(4 - np.sqrt(8))
    assert at_step_10.max() == pytest.approx(4 + np.sqrt(8))
    np.testing.assert_allclose(qac_band[qac_band[:, 0] == 0, 1], 1)

    gain = figures["predicted_vs_true_gain.png"].axes[0]
    kl = figures["update_kl_vs_improvement.png"].axes[0]
    rmse = figures["critic_rmse_vs_prediction_error.png"].axes[0]
    np.testing.assert_allclose(gain.collections[0].get_offsets(), [[1, 2], [6, 4], [3, 6], [8, 16]])
    np.testing.assert_allclose(kl.collections[0].get_offsets(), np.column_stack(([.01, .02, .03, .08],
                                                                             learned_gains)))
    np.testing.assert_allclose(rmse.collections[0].get_offsets(), [[3, 1], [6, 2], [9, 3], [24, 8]])
    for ax in (gain, kl, rmse):
        np.testing.assert_allclose(
            ax.collections[0].get_facecolors()[0, :3], lines[METHOD_LABELS["qac_full"]].get_color(),
        )
    comparison = figures["learned_vs_oracle_gain.png"].axes[0]
    np.testing.assert_allclose(comparison.collections[0].get_offsets(),
                               np.column_stack((learned_gains, [10, 20, 30, 40])))
    seed_gains = figures["learned_vs_oracle_gain.png"].axes[1]
    np.testing.assert_allclose(seed_gains.collections[0].get_offsets(),
                               [[np.mean(learned_gains[:3]), 20], [4, 40]])
    seed_kls = figures["update_kl_vs_improvement.png"].axes[1]
    np.testing.assert_allclose(seed_kls.collections[0].get_offsets(), [[.02, 9.99], [.08, 9.99]])
    oracle_kl = figures["update_kl_vs_improvement.png"].axes[0].collections[1]
    np.testing.assert_allclose(oracle_kl.get_offsets(), [[9.99, 40], [9.99, 30], [9.99, 20], [9.99, 10]])

    quality = figures["critic_quality_over_training.png"].axes
    np.testing.assert_allclose(quality[0].lines[0].get_xydata(), [[0, 0], [10, .51], [1000, 1]])
    np.testing.assert_allclose(quality[1].lines[0].get_ydata(), [.5, .55, .5])
    np.testing.assert_allclose(quality[2].lines[0].get_ydata(), [.2, .25, .2])
    alignment = figures["gradient_alignment_over_training.png"].axes
    np.testing.assert_allclose(alignment[0].lines[0].get_ydata(), [np.nan, .5, .5], equal_nan=True)
    np.testing.assert_allclose(alignment[1].lines[0].get_ydata(), [0, 1, 1])
    prediction = figures["prediction_error_over_training.png"].axes
    np.testing.assert_allclose(prediction[0].lines[0].get_xydata(), [[0, -1], [10, -3], [1000, -3]])
    np.testing.assert_allclose(prediction[1].lines[0].get_ydata(), [1, 5, 3])

    harmful = figures["harmful_update_frequency.png"].axes[0]
    # Seed 0: 1/2 harmful in the first bin, seed 1: 0/1. Equal seed weights give 1/4, not 1/3.
    np.testing.assert_allclose(harmful.lines[0].get_xydata(), [[500, .25], [1000, 0]])
    band = harmful.collections[0].get_paths()[0].vertices
    assert band[band[:, 0] == 500, 1].min() == pytest.approx(.25 - np.sqrt(.125))
    assert band[band[:, 0] == 500, 1].max() == pytest.approx(.25 + np.sqrt(.125))
    np.testing.assert_allclose(band[band[:, 0] == 1000, 1], 0)

    summary = pd.read_csv(output / "summary.csv")
    seeds = summary.loc[summary["statistic"] == "seed"].set_index("seed")
    metrics = [
        "final_expected_return", "mean_absolute_local_gain_prediction_error", "harmful_update_frequency",
        "mean_learned_guided_performance_gain", "mean_oracle_guided_performance_gain",
        "mean_paired_oracle_minus_learned_performance_gain",
    ]
    expected = np.array([
        [2, 2, 1 / 3, np.mean(learned_gains[:3]), 20, 20 - np.mean(learned_gains[:3])],
        [6, 8, 0, 4, 40, 36],
    ])
    np.testing.assert_allclose(seeds[metrics], expected)
    # Unequal checkpoint counts must not weight these seed-level aggregates.
    across = summary.set_index("statistic")
    np.testing.assert_allclose(across.loc["mean", metrics].astype(float), expected.mean(axis=0))
    np.testing.assert_allclose(across.loc["std", metrics].astype(float), expected.std(axis=0, ddof=1))
    np.testing.assert_allclose(seeds["mean_signed_local_gain_prediction_error"], [-2 / 3, -8])
    np.testing.assert_allclose(seeds["candidate_count"], [3, 1])
    counts = ["beneficial_update_count", "harmful_update_count", "effectively_zero_update_count"]
    np.testing.assert_allclose(seeds[counts], [[1, 1, 1], [1, 0, 0]])
    np.testing.assert_allclose(seeds[["beneficial_update_frequency", "effectively_zero_update_frequency"]],
                               [[1 / 3, 1 / 3], [1, 0]])
    np.testing.assert_allclose(seeds["mean_paired_oracle_minus_learned_state_kl"], [9.97, 9.91])
    np.testing.assert_allclose(seeds["defined_gradient_alignment_count"], [2, 1])
    np.testing.assert_allclose(seeds["learned_zero_gradient_count"], [1, 0])
    np.testing.assert_allclose(seeds["oracle_zero_gradient_count"], [0, 0])
    np.testing.assert_allclose(seeds["mean_gradient_cosine_similarity"], [.5, .5])
    all_metrics = seeds.columns.drop("statistic")
    np.testing.assert_allclose(across.loc["mean", all_metrics].astype(float), seeds[all_metrics].mean())
    np.testing.assert_allclose(across.loc["std", all_metrics].astype(float), seeds[all_metrics].std())


def test_fixed_training_step_bins():
    learned = pd.DataFrame({
        "seed": [0] * 5 + [1] * 2,
        "step": [0, 500, 501, 1000, 1001, 0, 1001],
        "harmful": [True, False, False, True, True, False, False],
    })
    learned["beneficial"] = ~learned["harmful"]
    learned["effectively_zero"] = False
    rates = update_rates(learned)
    np.testing.assert_allclose(rates[["seed", "bin_end", "harmful"]], [
        [0, 500, .5], [0, 1000, .5], [0, 1500, 1], [1, 500, 0], [1, 1500, 0],
    ])
    # Bin edges stay fixed when a later checkpoint is added.
    extended = pd.concat([learned, pd.DataFrame({"seed": [1], "step": [9000], "harmful": [True],
                                               "beneficial": [False], "effectively_zero": [False]})])
    pd.testing.assert_frame_equal(update_rates(extended).iloc[:-1], rates)


@pytest.mark.parametrize("key", ["training_method", "seed", "step", "state", "action", "method"])
def test_pairs_require_matching_provenance(key):
    learned = candidate(0, 0, "learned", 1)
    oracle = candidate(0, 0, "oracle", 2)
    oracle[key] = "other" if isinstance(oracle[key], str) else 1
    with pytest.raises(ValueError, match="both learned and oracle"):
        pair_candidates(pd.DataFrame([learned, oracle]))


def test_duplicate_pairs_rejected():
    learned = candidate(0, 0, "learned", 1)
    with pytest.raises(ValueError, match="duplicate"):
        pair_candidates(pd.DataFrame([learned, learned, candidate(0, 0, "oracle", 2)]))


def test_old_exports_require_regeneration(tmp_path, monkeypatch):
    write_csv(tmp_path / "checkpoints.csv", [checkpoint(0, 0, 1)])
    rows = [candidate(0, 0, source, 1) for source in ("learned", "oracle")]
    for row in rows:
        del row["gradient_cosine_similarity"]
    write_csv(tmp_path / "counterfactual_updates.csv", rows)
    monkeypatch.setattr(sys, "argv", ["policy-lab-plot", str(tmp_path)])
    with pytest.raises(ValueError, match="Missing gradient diagnostics; regenerate"):
        main()


def test_single_seed_default_output(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write_csv(data / "checkpoints.csv", [
        checkpoint(7, 0, 1),
    ])
    write_csv(data / "counterfactual_updates.csv", [
        candidate(7, 0, "learned", 1e-12), candidate(7, 0, "oracle", 2),
    ])
    monkeypatch.setattr(sys, "argv", ["policy-lab-plot", str(data)])
    main()
    output = tmp_path / "analysis"
    assert {path.name for path in output.iterdir()} == FIGURES | {"summary.csv"}
    summary = pd.read_csv(output / "summary.csv").set_index("statistic")
    metrics = summary.columns.drop("seed")
    np.testing.assert_allclose(summary.loc["seed", metrics], summary.loc["mean", metrics], equal_nan=True)
    assert summary.loc["std", metrics].isna().all()
    assert summary.loc["seed", "effectively_zero_update_count"] == 1
    assert summary.loc["seed", "effectively_zero_update_frequency"] == 1
    assert summary.loc["seed", "beneficial_update_count"] == 0
    assert summary.loc["seed", "defined_gradient_alignment_count"] == 0
    assert np.isnan(summary.loc["seed", "mean_gradient_cosine_similarity"])
