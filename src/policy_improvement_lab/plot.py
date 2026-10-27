"""Offline Experiment 001 analysis; never imported by the training code."""

import argparse
from pathlib import Path

from .diagnostics import GRADIENT_ZERO_TOLERANCE
from .mdp import make_branching_mdp

METHOD_LABELS = {
    "ac_td0": "One-step actor–critic",
    "qac_sampled": "Sampled-action QAC",
    "qac_full": "Full-action QAC",
    "qac_full_kl": "KL-backtracked full-action QAC",
}
GAIN_ZERO_TOLERANCE = 1e-12
TRAINING_STEP_BIN_WIDTH = 500
PAIR_KEYS = ["training_method", "seed", "step", "state", "action", "method"]


def pair_candidates(updates, value="actual_performance_gain"):
    """Require one learned/oracle pair at each frozen checkpoint candidate."""
    paired = updates.pivot(
        index=PAIR_KEYS, columns="critic_source", values=value,
    )
    if not {"learned", "oracle"}.issubset(paired.columns) or paired[["learned", "oracle"]].isna().any().any():
        raise ValueError("Each qac_full candidate must have both learned and oracle measurements")
    paired = paired.reset_index()
    paired["oracle_minus_learned"] = paired["oracle"] - paired["learned"]
    return paired


def summarize_seeds(checkpoints, learned, paired):
    """One row per seed; checkpoint counts never weight across-seed summaries."""
    final = checkpoints.sort_values("step").groupby("seed").tail(1).set_index("seed")
    summary = final[["expected_return"]].rename(columns={"expected_return": "final_expected_return"})
    summary["mean_absolute_local_gain_prediction_error"] = learned.groupby("seed")[
        "absolute_local_gain_prediction_error"
    ].mean()
    summary["mean_signed_local_gain_prediction_error"] = learned.groupby("seed")[
        "signed_local_gain_prediction_error"
    ].mean()
    summary["candidate_count"] = learned.groupby("seed").size()
    for outcome in ("beneficial", "harmful", "effectively_zero"):
        summary[f"{outcome}_update_count"] = learned.groupby("seed")[outcome].sum()
        summary[f"{outcome}_update_frequency"] = learned.groupby("seed")[outcome].mean()
    for source in ("learned", "oracle"):
        summary[f"mean_{source}_guided_state_kl"] = learned.groupby("seed")[f"{source}_state_kl"].mean()
        summary[f"{source}_zero_gradient_count"] = learned.groupby("seed")[
            f"{source}_gradient_norm"
        ].agg(lambda values: (values <= GRADIENT_ZERO_TOLERANCE).sum())
    summary["mean_paired_oracle_minus_learned_state_kl"] = (
        summary["mean_oracle_guided_state_kl"] - summary["mean_learned_guided_state_kl"]
    )
    summary["mean_learned_guided_performance_gain"] = paired.groupby("seed")["learned"].mean()
    summary["mean_oracle_guided_performance_gain"] = paired.groupby("seed")["oracle"].mean()
    summary["mean_paired_oracle_minus_learned_performance_gain"] = paired.groupby("seed")[
        "oracle_minus_learned"
    ].mean()
    if set(summary.index) != set(learned["seed"]) or summary.isna().any().any():
        raise ValueError("Every qac_full training seed must have candidate measurements")
    # Undefined cosines are excluded, never interpreted as zero alignment.
    alignment = learned.groupby("seed")["gradient_cosine_similarity"]
    summary["defined_gradient_alignment_count"] = alignment.count()
    summary["mean_gradient_cosine_similarity"] = alignment.mean()
    return summary.sort_index()


def update_rates(learned):
    """Seed-level frequencies in [0, 500], (500, 1000], ... step bins."""
    rows = learned.assign(
        bin_end=((learned["step"] - 1).clip(lower=0) // TRAINING_STEP_BIN_WIDTH + 1)
        * TRAINING_STEP_BIN_WIDTH,
    )
    outcomes = ["beneficial", "harmful", "effectively_zero"]
    return rows.groupby(["seed", "bin_end"])[outcomes].mean().reset_index()


def plot_seed_curve(ax, rows, x, y, *, color, label=None):
    """Equal seed weights and sample SD; repeated checkpoints are not replicates."""
    per_seed = rows.groupby(["seed", x])[y].mean().reset_index()
    summary = per_seed.groupby(x)[y].agg(mean="mean", std="std").reset_index()
    summary["std"] = summary["std"].fillna(0)
    ax.plot(summary[x], summary["mean"], color=color, label=label)
    ax.fill_between(
        summary[x], summary["mean"] - summary["std"], summary["mean"] + summary["std"],
        alpha=0.12, color=color,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 001 qac_full outputs")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, help="Output directory (default: sibling analysis directory)")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise SystemExit("Plotting requires: uv sync --extra analysis") from exc

    checkpoints = pd.read_csv(args.results_dir / "checkpoints.csv")
    updates = pd.read_csv(args.results_dir / "counterfactual_updates.csv")
    checkpoints = checkpoints.loc[checkpoints["method"] == "qac_full"].copy()
    updates = updates.loc[
        (updates["training_method"] == "qac_full") & (updates["method"] == "qac_full")
        & updates["critic_source"].isin(["learned", "oracle"])
    ].copy()
    if checkpoints.empty or updates.empty:
        raise ValueError("Experiment 001 requires qac_full checkpoints and qac_full candidates")
    gradient_columns = {"learned_gradient_norm", "oracle_gradient_norm", "gradient_cosine_similarity"}
    if not gradient_columns.issubset(updates.columns):
        raise ValueError("Missing gradient diagnostics; regenerate the run with the current exporter")
    # Duplicate checkpoints would overweight a seed in the training curve.
    if checkpoints.duplicated(["seed", "step"]).any():
        raise ValueError("Duplicate qac_full checkpoints for the same seed and step")
    paired = pair_candidates(updates)
    learned = updates.loc[updates["critic_source"] == "learned"].copy()
    learned["signed_local_gain_prediction_error"] = (
        learned["predicted_local_gain"] - learned["true_local_gain"]
    )
    learned["absolute_local_gain_prediction_error"] = learned["signed_local_gain_prediction_error"].abs()
    learned["beneficial"] = learned["actual_performance_gain"] > GAIN_ZERO_TOLERANCE
    learned["harmful"] = learned["actual_performance_gain"] < -GAIN_ZERO_TOLERANCE
    learned["effectively_zero"] = learned["actual_performance_gain"].abs() <= GAIN_ZERO_TOLERANCE
    paired_kl = pair_candidates(updates, "state_kl")
    learned = learned.merge(
        paired_kl[PAIR_KEYS + ["learned", "oracle"]].rename(
            columns={"learned": "learned_state_kl", "oracle": "oracle_state_kl"},
        ), on=PAIR_KEYS, validate="one_to_one",
    )
    seed_summary = summarize_seeds(checkpoints, learned, paired)
    # Sample SD is undefined with one seed; leave it missing in the CSV.
    across_seeds = seed_summary.agg(["mean", "std"]).rename_axis("statistic").reset_index()
    per_seed = seed_summary.reset_index().assign(statistic="seed")

    output = args.out_dir or args.results_dir.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat([per_seed, across_seeds], ignore_index=True).to_csv(output / "summary.csv", index=False)
    sns.set_theme(style="whitegrid", context="notebook")
    color = sns.color_palette("colorblind", n_colors=len(METHOD_LABELS))[2]

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_seed_curve(
        ax, checkpoints, "step", "expected_return", color=color, label=METHOD_LABELS["qac_full"],
    )
    ax.axhline(make_branching_mdp().optimal_performance(), color="black", linestyle="--", label="Optimal J*")
    ax.set(xlabel="Environment steps", ylabel="Exact expected discounted return", title="Training curves")
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=learned, x="predicted_local_gain", y="true_local_gain",
        s=10, alpha=0.40, linewidth=0, color=color, ax=ax,
    )
    bounds = (min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot(bounds, bounds, linestyle="--", linewidth=1, color="black", alpha=0.7)
    ax.set(
        xlabel="Predicted local gain", ylabel="True local gain", title="Critic predicted vs true local gain"
    )
    fig.tight_layout()
    fig.savefig(output / "predicted_vs_true_gain.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    oracle_color = sns.color_palette("colorblind")[1]
    for source, source_color in (("learned", color), ("oracle", oracle_color)):
        sns.scatterplot(
            data=updates.loc[updates["critic_source"] == source], x="state_kl", y="actual_performance_gain",
            s=10, alpha=0.40, linewidth=0, color=source_color, label=source, ax=ax,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set(
        xlabel="Realized state KL: D_KL(new || old)",
        ylabel="Exact change in J",
        title="Update magnitude vs quality (checkpoints)",
    )
    ax.legend(title="Guidance")
    ax.locator_params(axis="x", nbins=5)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=False)
    ax = axes[1]
    sns.scatterplot(
        data=seed_summary, x="mean_learned_guided_state_kl", y="mean_oracle_guided_state_kl",
        color=color, ax=ax,
    )
    bounds = (0, max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot(bounds, bounds, color="black", linestyle=":", linewidth=1)
    for seed, row in seed_summary.iterrows():
        ax.annotate(str(seed), (row["mean_learned_guided_state_kl"], row["mean_oracle_guided_state_kl"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean state KL: learned", ylabel="Mean state KL: oracle",
           title="Paired seed means (seed labels)")
    ax.locator_params(axis="both", nbins=5)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useOffset=False)
    fig.tight_layout()
    fig.savefig(output / "update_kl_vs_improvement.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    sns.scatterplot(
        data=paired, x="learned", y="oracle", color=color,
        s=15, alpha=0.5, linewidth=0, ax=ax,
    )
    bounds = (min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot(bounds, bounds, color="black", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set(
        xlabel="Exact change in J: learned-guided update",
        ylabel="Exact change in J: oracle-guided update",
        title="Paired full-action candidates at the same checkpoint",
    )
    ax = axes[1]
    sns.scatterplot(
        data=seed_summary, x="mean_learned_guided_performance_gain", y="mean_oracle_guided_performance_gain",
        color=color, ax=ax,
    )
    bounds = (min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot(bounds, bounds, color="black", linestyle=":", linewidth=1)
    for seed, row in seed_summary.iterrows():
        ax.annotate(str(seed), (row["mean_learned_guided_performance_gain"],
                               row["mean_oracle_guided_performance_gain"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean change in J: learned", ylabel="Mean change in J: oracle",
           title="Paired seed means (seed labels)")
    ax.locator_params(axis="both", nbins=5)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useOffset=False)
    fig.tight_layout()
    fig.savefig(output / "learned_vs_oracle_gain.png", dpi=160)
    plt.close(fig)

    rates = update_rates(learned)
    fig, ax = plt.subplots(figsize=(9, 5))
    for outcome, outcome_color in (
        ("harmful", "#D55E00"), ("beneficial", color), ("effectively_zero", "gray"),
    ):
        plot_seed_curve(ax, rates, "bin_end", outcome, color=outcome_color, label=outcome.replace("_", " "))
    ax.set(
        xlabel="Environment steps (500-step bin end)", ylabel="Update frequency", ylim=(-0.05, 1.05),
        title="Learned-guided update outcomes (mean ± SD across seeds)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "harmful_update_frequency.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=learned, x="rmse", y="absolute_local_gain_prediction_error",
        s=10, alpha=0.40, linewidth=0, color=color, ax=ax,
    )
    ax.set(
        xlabel="Learned critic RMSE", ylabel="Absolute local-gain prediction error",
        title="Critic error vs gain prediction error (learned guidance)",
    )
    fig.tight_layout()
    fig.savefig(output / "critic_rmse_vs_prediction_error.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, label in zip(
        axes, ("rmse", "ranking_accuracy", "consequential_ranking_regret"),
        ("Critic RMSE", "Action-ranking accuracy", "Action-selection regret"), strict=True,
    ):
        plot_seed_curve(ax, checkpoints, "step", metric, color=color)
        ax.set(xlabel="Environment steps", ylabel=label)
    fig.suptitle("Learned critic quality (mean ± SD across seeds)")
    fig.tight_layout()
    fig.savefig(output / "critic_quality_over_training.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_seed_curve(ax, learned, "step", "gradient_cosine_similarity", color=color)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Environment steps", ylabel="Gradient cosine similarity", ylim=(-1.05, 1.05))
    fig.suptitle("Full-action gradient alignment\n(mean ± SD; undefined cosines excluded)")
    fig.tight_layout()
    fig.savefig(output / "gradient_alignment_over_training.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_seed_curve(ax, learned, "step", "signed_local_gain_prediction_error", color=color)
    ax.set(xlabel="Environment steps", ylabel="Signed local-gain error")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("+ overestimate / − underestimate")
    fig.suptitle("Learned-guided gain prediction error (mean ± SD across seeds)")
    fig.tight_layout()
    fig.savefig(output / "prediction_error_over_training.png", dpi=160)
    plt.close(fig)
    print(f"Analysis saved in {output.resolve()}")


if __name__ == "__main__":
    main()
