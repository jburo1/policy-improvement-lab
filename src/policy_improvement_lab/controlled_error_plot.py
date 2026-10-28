"""Offline EXP002 pairing, equal-seed summaries, and noise-response figures."""

import argparse
import json
from pathlib import Path

import numpy as np

from .diagnostics import GRADIENT_ZERO_TOLERANCE
from .plot import plot_seed_curve

CHECKPOINT_KEYS = ["run_id", "seed", "step", "state"]
CONDITION_KEYS = ["condition", "magnitude"]
MEASUREMENTS = [
    "checkpoint_expected_return",
    "local_rmse", "local_ranking_accuracy", "local_gap_scale", "local_noise_sigma",
    "guidance_gradient_norm", "oracle_gradient_norm", "gradient_cosine_similarity",
    "true_local_gain", "actual_performance_gain", "state_kl",
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
    if not (checkpoints.method.eq("qac_full").all() and config["methods"] == ["qac_full"]):
        raise ValueError("EXP002 supports only full-action QAC")
    steps = config["run_config"]["steps"]
    every = config["run_config"]["eval_every"]
    expected_steps = sorted({0, steps, *range(every, steps + 1, every)})
    for _, group in checkpoints.groupby("seed"):
        if sorted(group.step) != expected_steps:
            raise ValueError("Missing or duplicate training checkpoints")
    cases = [("oracle", 0.0, -1), ("reversed_guidance", 1.0, -1)]
    cases += [("additive_bias", config["additive_bias"], -1)]
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
    return summary.reset_index()


def summarize_seeds(checkpoint_summary):
    keys = ["run_id", "seed", *CONDITION_KEYS]
    groups = checkpoint_summary.groupby(keys, dropna=False)
    result = groups[MEASUREMENTS].mean()
    result["checkpoint_count"] = groups.size()
    result[COUNTS] = groups[COUNTS].sum()
    return result.reset_index()


def summarize_across_seeds(seed_summary):
    """Equal seed weights; SD is sample SD and undefined for a single seed."""
    keys = ["run_id", *CONDITION_KEYS]
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
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("checkpoint_summary", checkpoint_summary), ("seed_summary", seeds),
        ("summary", summarize_across_seeds(seeds)),
    ):
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    # Exact guidance is the sole zero-noise reference, not another set of draws.
    noise = seeds.loc[seeds.condition.isin(["gaussian_noise", "oracle"])]
    ticks = [0.0, *config["noise_multipliers"]]
    for ax, metric, label in zip(axes, ["harmful_frequency", "actual_performance_gain"],
                                 ["Harmful update frequency", "Mean exact improvement ΔJ"], strict=True):
        plot_seed_curve(ax, noise, "magnitude", metric, color="tab:blue")
        ax.set(xlabel="Noise multiplier (σ / local gap scale)", ylabel=label)
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xticks(ticks, [f"{m:g}" for m in ticks])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(alpha=0.2)
    axes[0].set_ylim(-0.05, 1.05)
    fig.suptitle("Noise response: equal seed weights, mean ± seed SD; 0 = oracle")
    fig.savefig(out_dir / "noise_response.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    panels = [
        ("gradient_cosine_similarity", "Local gradient cosine similarity"),
        ("state_kl", "Realized state KL"),
    ]
    for ax, (x, label) in zip(axes, panels, strict=True):
        for condition, rows in checkpoint_summary.groupby("condition"):
            ax.scatter(rows[x], rows.actual_performance_gain, s=12, alpha=0.4, label=condition)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set(xlabel=label, ylabel="Exact global improvement ΔJ")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Descriptive checkpoint means (noise draws averaged within checkpoint)")
    fig.savefig(out_dir / "gradient_alignment_vs_improvement.png", dpi=160)
    plt.close(fig)
    print(f"Summaries and two figures written to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    analyze(args.results_dir, args.out_dir or args.results_dir.parent / "analysis")


if __name__ == "__main__":
    main()
