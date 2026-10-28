"""Offline EXP002 pairing, hierarchical summaries, and the three planned analyses."""

import argparse
import json
from pathlib import Path

import numpy as np

from .diagnostics import GRADIENT_ZERO_TOLERANCE
from .plot import TRAINING_STEP_BIN_WIDTH, plot_seed_curve

CHECKPOINT_KEYS = ["run_id", "seed", "step", "state"]
CONDITION_KEYS = ["condition", "magnitude"]
MEASUREMENTS = [
    "checkpoint_expected_return", "rmse", "ranking_accuracy", "consequential_ranking_regret",
    "local_rmse", "local_ranking_accuracy", "local_gap_scale", "local_noise_sigma",
    "guidance_gradient_norm", "oracle_gradient_norm", "gradient_cosine_similarity",
    "predicted_local_gain", "true_local_gain", "actual_performance_gain", "state_kl",
    "oracle_performance_gain", "oracle_minus_perturbed_gain",
    "beneficial_frequency", "harmful_frequency", "effectively_zero_frequency",
    "defined_alignment_frequency", "guidance_zero_gradient_frequency", "oracle_zero_gradient_frequency",
]
COUNTS = ["candidate_count", "beneficial_count", "harmful_count", "effectively_zero_count",
          "defined_alignment_count", "guidance_zero_gradient_count", "oracle_zero_gradient_count"]


def validate_pairs(checkpoints, updates, config):
    """Reject incomplete/duplicate candidate grids and broken checkpoint/oracle pairing."""
    import pandas as pd

    if checkpoints.empty or updates.empty:
        raise ValueError("EXP002 requires checkpoints and candidate measurements")
    if (set(checkpoints.run_id) != {config["run_id"]} or set(updates.run_id) != {config["run_id"]}
            or set(checkpoints.seed) != set(config["seeds"])):
        raise ValueError("Run IDs and seeds must match config.json")
    if not (checkpoints.method.eq("qac_full").all() and updates.method.eq("qac_full").all()
            and updates.training_method.eq("qac_full").all()):
        raise ValueError("EXP002 supports only full-action QAC")
    steps = config["run_config"]["steps"]
    every = config["run_config"]["eval_every"]
    expected_steps = sorted({0, steps, *range(every, steps + 1, every)})
    for _, group in checkpoints.groupby("seed"):
        if sorted(group.step) != expected_steps:
            raise ValueError("Missing or duplicate training checkpoints")
    cases = [("oracle", 0.0, -1), ("reversed_guidance", 1.0, -1)]
    cases += [("additive_bias", b, -1) for b in config["additive_biases"]]
    cases += [("gaussian_noise", m, r) for m in config["noise_multipliers"]
              for r in range(config["noise_replicates"])]
    expected = pd.MultiIndex.from_tuples(
        [(*key, *case) for key in checkpoints[CHECKPOINT_KEYS].itertuples(index=False, name=None)
         for case in cases], names=CHECKPOINT_KEYS + CONDITION_KEYS + ["replicate"],
    ).sort_values()
    actual = pd.MultiIndex.from_frame(updates[CHECKPOINT_KEYS + CONDITION_KEYS + ["replicate"]]).sort_values()
    if not actual.equals(expected):
        raise ValueError("Missing, duplicate, or mismatched checkpoint candidates/replicates")
    paired = updates.merge(
        checkpoints[CHECKPOINT_KEYS + ["expected_return"]], on=CHECKPOINT_KEYS, validate="many_to_one",
    )
    if not np.allclose(paired.checkpoint_expected_return, paired.expected_return, rtol=0, atol=1e-14):
        raise ValueError("Candidate return differs from its frozen checkpoint")
    oracle = updates.loc[updates.condition.eq("oracle"), CHECKPOINT_KEYS + ["actual_performance_gain"]]
    paired = updates.merge(oracle, on=CHECKPOINT_KEYS, suffixes=("", "_reference"), validate="many_to_one")
    if not np.allclose(paired.oracle_performance_gain, paired.actual_performance_gain_reference,
                       rtol=0, atol=1e-14):
        raise ValueError("Oracle comparison must use the same frozen checkpoint")
    if not np.allclose(paired.oracle_minus_perturbed_gain,
                       paired.actual_performance_gain_reference - paired.actual_performance_gain,
                       rtol=0, atol=1e-14):
        raise ValueError("Incorrect paired oracle-minus-perturbed difference")


def summarize_checkpoints(updates):
    """Average draws first; each resulting row has one frozen-checkpoint weight."""
    rows = updates.copy()
    for outcome in ("beneficial", "harmful", "effectively_zero"):
        rows[f"{outcome}_frequency"] = rows[outcome]
    rows["defined_alignment_frequency"] = rows.gradient_cosine_similarity.notna().astype(int)
    rows["guidance_zero_gradient_frequency"] = (
        rows.guidance_gradient_norm <= GRADIENT_ZERO_TOLERANCE
    ).astype(int)
    rows["oracle_zero_gradient_frequency"] = (
        rows.oracle_gradient_norm <= GRADIENT_ZERO_TOLERANCE
    ).astype(int)
    keys = CHECKPOINT_KEYS + CONDITION_KEYS
    grouped = rows.groupby(keys, dropna=False)
    summary = grouped[MEASUREMENTS].mean()
    summary["candidate_count"] = grouped.size()
    for name in ("beneficial", "harmful", "effectively_zero", "defined_alignment",
                 "guidance_zero_gradient", "oracle_zero_gradient"):
        summary[f"{name}_count"] = grouped[f"{name}_frequency"].sum()
    summary = summary.reset_index()
    summary["bin_end"] = ((summary.step - 1).clip(lower=0) // TRAINING_STEP_BIN_WIDTH + 1) \
        * TRAINING_STEP_BIN_WIDTH
    return summary


def summarize_seeds(checkpoint_summary, *, by_stage=False):
    keys = ["run_id", "seed", *CONDITION_KEYS] + (["bin_end"] if by_stage else [])
    groups = checkpoint_summary.groupby(keys, dropna=False)
    result = groups[MEASUREMENTS].mean()
    result["checkpoint_count"] = groups.size()
    result[COUNTS] = groups[COUNTS].sum()
    return result.reset_index()


def summarize_across_seeds(seed_summary, *, by_stage=False):
    """Equal seed weights; SD is sample SD and undefined for a single seed."""
    keys = ["run_id", *CONDITION_KEYS] + (["bin_end"] if by_stage else [])
    metrics = MEASUREMENTS + ["checkpoint_count", *COUNTS]
    result = seed_summary.groupby(keys)[metrics].agg(["mean", "std", "count"])
    result.columns = [f"{metric}_{statistic}" for metric, statistic in result.columns]
    return result.reset_index()


def analyze(results_dir: Path, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    config = json.loads((results_dir / "config.json").read_text(encoding="utf-8"))
    checkpoints = pd.read_csv(results_dir / "checkpoints.csv")
    updates = pd.read_csv(results_dir / "counterfactual_updates.csv")
    validate_pairs(checkpoints, updates, config)
    checkpoint_summary = summarize_checkpoints(updates)
    seeds = summarize_seeds(checkpoint_summary)
    stages = summarize_seeds(checkpoint_summary, by_stage=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("checkpoint_summary", checkpoint_summary), ("seed_summary", seeds),
        ("summary", summarize_across_seeds(seeds)), ("stage_seed_summary", stages),
        ("stage_summary", summarize_across_seeds(stages, by_stage=True)),
    ):
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    noise = seeds.loc[seeds.condition.eq("gaussian_noise")]
    plot_seed_curve(axes[0], noise, "magnitude", "harmful_frequency", color="tab:blue")
    axes[0].set_title("All checkpoints: equal seed weights, mean ± SD")
    stage_noise = stages.loc[stages.condition.eq("gaussian_noise")]
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, stage_noise.bin_end.nunique()))
    for (end, rows), color in zip(stage_noise.groupby("bin_end"), colors, strict=True):
        plot_seed_curve(axes[1], rows, "magnitude", "harmful_frequency", color=color,
                        label=f"{max(0, end - TRAINING_STEP_BIN_WIDTH)}–{end} steps")
    axes[1].set_title("By training stage: mean ± SD across seeds")
    for ax in axes:
        ax.set(xlabel="Noise multiplier (σ / local gap scale)", ylabel="Harmful update frequency")
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xticks(config["noise_multipliers"], [f"{m:g}" for m in config["noise_multipliers"]])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.savefig(out_dir / "noise_magnitude_vs_harmful_frequency.png", dpi=160)
    plt.close(fig)

    def scatter_figure(filename, panels):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, (x, label) in zip(axes, panels, strict=True):
            for condition, rows in checkpoint_summary.groupby("condition"):
                ax.scatter(rows[x], rows.actual_performance_gain, s=12, alpha=0.4, label=condition)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set(xlabel=label, ylabel="Exact global improvement ΔJ")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("Descriptive checkpoint means (noise draws averaged within checkpoint)")
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    scatter_figure("gradient_alignment_vs_improvement.png", [
        ("gradient_cosine_similarity", "Local gradient cosine similarity"),
        ("state_kl", "Realized state KL"),
    ])
    scatter_figure("critic_rmse_vs_improvement.png", [
        ("rmse", "Global critic RMSE"), ("local_rmse", "State-local critic RMSE"),
    ])
    print(f"Summaries and three figures written to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    analyze(args.results_dir, args.out_dir or args.results_dir.parent / "analysis")


if __name__ == "__main__":
    main()
