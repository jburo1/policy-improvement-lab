"""Offline Pandas/Seaborn analysis; never imported by the training code."""

import argparse
from pathlib import Path

METHOD_LABELS = {
    "ac_td0": "One-step actor–critic",
    "qac_sampled": "Sampled-action QAC",
    "qac_full": "Full-action QAC",
    "qac_full_kl": "KL-backtracked full-action QAC",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot policy-lab experiment outputs")
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise SystemExit("Plotting requires: uv sync --extra analysis") from exc

    sns.set_theme(style="whitegrid", context="notebook")
    colors = dict(zip(
        METHOD_LABELS, sns.color_palette("colorblind", n_colors=len(METHOD_LABELS)), strict=True,
    ))
    output = args.results_dir
    checkpoints = pd.read_csv(output / "checkpoints.csv")
    updates = pd.read_csv(output / "counterfactual_updates.csv")
    summary = (
        checkpoints.groupby(["method", "step"])["expected_return"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    # Pandas uses sample SD (ddof=1); retain a zero-width band for a single seed.
    summary["std"] = summary["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, label in METHOD_LABELS.items():
        rows = summary.loc[summary["method"] == method]
        if rows.empty:
            continue
        sns.lineplot(
            data=rows, x="step", y="mean", estimator=None, errorbar=None,
            label=label, color=colors[method], ax=ax,
        )
        ax.fill_between(
            rows["step"], rows["mean"] - rows["std"], rows["mean"] + rows["std"],
            alpha=0.12, color=colors[method],
        )
    ax.set(xlabel="Environment steps", ylabel="Exact expected discounted return", title="Training curves")
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)

    learned = updates.loc[updates["critic_source"] == "learned"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for method in ("qac_sampled", "qac_full", "qac_full_kl"):
        rows = learned.loc[learned["method"] == method]
        if not rows.empty:
            sns.scatterplot(
                data=rows, x="predicted_local_gain", y="true_local_gain",
                s=10, alpha=0.40, linewidth=0, label=METHOD_LABELS[method], color=colors[method], ax=ax,
            )
    bounds = ax.get_xlim()
    ax.plot(bounds, bounds, linestyle="--", linewidth=1, color="black", alpha=0.7)
    ax.set(
        xlabel="Predicted local gain", ylabel="True local gain", title="Critic predicted vs true local gain"
    )
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(output / "predicted_vs_true_gain.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for method in ("qac_sampled", "qac_full", "qac_full_kl"):
        rows = learned.loc[learned["method"] == method]
        if not rows.empty:
            sns.scatterplot(
                data=rows, x="state_kl", y="actual_performance_gain",
                s=10, alpha=0.40, linewidth=0, label=METHOD_LABELS[method], color=colors[method], ax=ax,
            )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set(
        xlabel="Realized state KL: D_KL(new || old)",
        ylabel="Exact change in J",
        title="Update size vs improvement",
    )
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(output / "update_kl_vs_improvement.png", dpi=160)
    plt.close(fig)
    print(f"Plots saved in {output.resolve()}")


if __name__ == "__main__":
    main()
