"""Command-line entry point for reproducible, seeded tabular experiments."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .mdp import make_branching_mdp
from .operators import ALL_METHODS
from .training import RunConfig, train


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--actor-lr", type=float, default=0.20)
    parser.add_argument("--critic-lr", type=float, default=0.12)
    parser.add_argument("--kl-limit", type=float, default=0.002)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/first-run"))
    args = parser.parse_args()
    cfg = RunConfig(args.steps, args.eval_every, args.actor_lr, args.critic_lr, args.kl_limit)
    mdp = make_branching_mdp()
    checkpoints: list[dict] = []
    updates: list[dict] = []
    for method in args.methods:
        for seed in args.seeds:
            history, candidates = train(mdp, method, seed, cfg)
            checkpoints.extend(history)
            updates.extend(candidates)
            print(f"{method:14s} seed={seed:3d} J(0)={history[0]['expected_return']:.3f} "
                  f"J(end)={history[-1]['expected_return']:.3f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "checkpoints.csv", checkpoints)
    write_csv(args.out_dir / "counterfactual_updates.csv", updates)
    (args.out_dir / "config.json").write_text(
        json.dumps({"run_config": asdict(cfg), "methods": args.methods, "seeds": args.seeds,
                    "environment": "branching_mdp_v1", "implementation": "tabular"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Records written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
