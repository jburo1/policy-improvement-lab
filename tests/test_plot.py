"""Protect the plot statistics and learned-critic filtering."""

import sys

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("seaborn")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from policy_improvement_lab.cli import write_csv  # noqa: E402
from policy_improvement_lab.plot import METHOD_LABELS, main  # noqa: E402


def test_plot_statistics_and_candidate_filtering(tmp_path, monkeypatch):
    write_csv(tmp_path / "checkpoints.csv", [
        {"method": "qac_full", "step": 10, "seed": 1, "expected_return": 6},
        {"method": "qac_full", "step": 0, "seed": 0, "expected_return": 1},
        {"method": "qac_full", "step": 10, "seed": 0, "expected_return": 2},
        {"method": "ac_td0", "step": 0, "seed": 0, "expected_return": 3},
    ])
    write_csv(tmp_path / "counterfactual_updates.csv", [
        {"method": "qac_full", "critic_source": source, "predicted_local_gain": value,
         "true_local_gain": value * 2, "state_kl": value / 100,
         "actual_performance_gain": -value}
        for source, value in (("learned", 1), ("learned", 2), ("oracle", 99))
    ])
    figures = {}
    savefig = Figure.savefig

    def capture(fig, path, **kwargs):
        figures[path.name] = fig
        savefig(fig, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    monkeypatch.setattr(sys, "argv", ["policy-lab-plot", str(tmp_path)])
    main()

    assert set(figures) == {
        "learning_curves.png", "predicted_vs_true_gain.png", "update_kl_vs_improvement.png",
    }
    for name in figures:
        assert (tmp_path / name).stat().st_size > 0

    curves = figures["learning_curves.png"].axes[0]
    lines = {line.get_label(): line for line in curves.lines}
    np.testing.assert_allclose(lines[METHOD_LABELS["qac_full"]].get_xydata(), [[0, 1], [10, 4]])
    np.testing.assert_allclose(lines[METHOD_LABELS["ac_td0"]].get_xydata(), [[0, 3]])
    # Check the plotted band itself: sample SD is sqrt(8), while a singleton has SD zero.
    qac_band = curves.collections[1].get_paths()[0].vertices
    at_step_10 = qac_band[qac_band[:, 0] == 10, 1]
    assert at_step_10.min() == pytest.approx(4 - np.sqrt(8))
    assert at_step_10.max() == pytest.approx(4 + np.sqrt(8))
    np.testing.assert_allclose(qac_band[qac_band[:, 0] == 0, 1], 1)
    np.testing.assert_allclose(curves.collections[0].get_paths()[0].vertices[:, 1], 3)

    gain = figures["predicted_vs_true_gain.png"].axes[0]
    kl = figures["update_kl_vs_improvement.png"].axes[0]
    np.testing.assert_allclose(gain.collections[0].get_offsets(), [[1, 2], [2, 4]])
    np.testing.assert_allclose(kl.collections[0].get_offsets(), [[0.01, -1], [0.02, -2]])
    for ax in (gain, kl):
        np.testing.assert_allclose(
            ax.collections[0].get_facecolors()[0, :3], lines[METHOD_LABELS["qac_full"]].get_color(),
        )
