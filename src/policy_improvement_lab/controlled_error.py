"""EXP002 frozen-checkpoint controls, using the unchanged EXP001 learner."""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .cli import write_csv
from .diagnostics import critic_quality, gradient_alignment
from .mdp import FiniteMDP, make_branching_mdp
from .operators import kl_new_old, softmax, update_logits
from .training import RunConfig, train

NOISE_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ADDITIVE_BIASES = (0.0, 1.0, 10.0)
NOISE_REPLICATES = 20
GAP_FLOOR = 1e-6
GAIN_ZERO_TOLERANCE = 1e-12


def gap_scales(q: np.ndarray) -> np.ndarray:
    """Per-state RMS over unordered action-pair gaps, floored in value units."""
    a, b = np.triu_indices(q.shape[1], k=1)
    return np.maximum(np.sqrt(np.mean((q[:, a] - q[:, b]) ** 2, axis=1)), GAP_FLOOR)


def noise_seed(seed: int, step: int, magnitude: float, replicate: int, root: int = 0) -> int:
    """Stable keys independent of condition/grid iteration order or training RNG."""
    bits = int(np.asarray(float(magnitude), dtype=np.float64).view(np.uint64))
    sequence = np.random.SeedSequence([2002, root, seed, step, bits & 0xFFFFFFFF, bits >> 32, replicate])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def construct_guidance(
    mdp: FiniteMDP, exact_q: np.ndarray, condition: str, magnitude: float = 0.0,
    *, sample_seed: int | None = None,
) -> np.ndarray:
    """Construct a new table; perturb only decision states, including unvisited ones."""
    guidance = exact_q.copy()
    active = mdp.active_states
    if condition == "additive_bias":
        guidance[active] += magnitude
    elif condition == "gaussian_noise":
        if sample_seed is None:
            raise ValueError("Gaussian noise requires an explicit sample seed")
        noise = np.random.default_rng(sample_seed).standard_normal((len(active), mdp.n_actions))
        guidance[active] += magnitude * gap_scales(exact_q)[active, None] * noise
    elif condition == "reversed_guidance":
        guidance[active] *= -1
    elif condition != "oracle":
        raise ValueError(f"Unknown condition: {condition}")
    return guidance


def evaluate_candidate(
    mdp: FiniteMDP, logits: np.ndarray, exact_q: np.ndarray, state: int,
    guidance: np.ndarray, cfg: RunConfig,
) -> dict:
    """One independent full-action update and exact diagnostics at the frozen state."""
    old_pi = softmax(logits)
    old_j = mdp.performance(old_pi)
    # Full-action QAC ignores the action argument and does not backtrack on KL.
    new_logits = update_logits(logits, state, "qac_full", 0, guidance, cfg.actor_lr, cfg.kl_limit)
    new_pi = softmax(new_logits)
    delta = new_pi[state] - old_pi[state]
    quality = critic_quality(mdp, guidance, exact_q)
    local = critic_quality(mdp, guidance, exact_q, states=np.array([state]))
    alignment = asdict(gradient_alignment(old_pi[state], guidance[state], exact_q[state]))
    alignment["guidance_gradient_norm"] = alignment.pop("learned_gradient_norm")
    gain = mdp.performance(new_pi) - old_j
    return {
        "checkpoint_expected_return": old_j,
        **asdict(quality), "local_rmse": local.rmse, "local_ranking_accuracy": local.ranking_accuracy,
        **alignment,
        "predicted_local_gain": float(delta @ guidance[state]),
        "true_local_gain": float(delta @ exact_q[state]),
        "actual_performance_gain": gain,
        "state_kl": kl_new_old(new_pi[state], old_pi[state]),
        "beneficial": int(gain > GAIN_ZERO_TOLERANCE),
        "harmful": int(gain < -GAIN_ZERO_TOLERANCE),
        "effectively_zero": int(abs(gain) <= GAIN_ZERO_TOLERANCE),
    }


def checkpoint_candidates(
    mdp: FiniteMDP, logits: np.ndarray, state: int, seed: int, step: int, cfg: RunConfig,
    *, noise_root: int = 0,
) -> list[dict]:
    """All candidates start from the supplied logits; none are committed to training."""
    _, exact_q = mdp.exact_values(softmax(logits))
    oracle = evaluate_candidate(mdp, logits, exact_q, state, exact_q, cfg)
    cases = [("oracle", 0.0, -1), ("reversed_guidance", 1.0, -1)]
    cases += [("additive_bias", bias, -1) for bias in ADDITIVE_BIASES]
    cases += [("gaussian_noise", scale, rep)
              for scale in NOISE_MULTIPLIERS for rep in range(NOISE_REPLICATES)]
    rows = []
    scale = float(gap_scales(exact_q)[state])
    for condition, magnitude, replicate in cases:
        sample_seed = noise_seed(seed, step, magnitude, replicate, noise_root) if replicate >= 0 else None
        guidance = construct_guidance(mdp, exact_q, condition, magnitude, sample_seed=sample_seed)
        metrics = evaluate_candidate(mdp, logits, exact_q, state, guidance, cfg)
        rows.append({
            "seed": seed, "step": step, "state": state,
            "training_method": "qac_full", "method": "qac_full",
            "condition": condition, "magnitude": magnitude,
            "replicate": replicate, "noise_seed": "" if sample_seed is None else f"0x{sample_seed:016x}",
            "local_gap_scale": scale,
            "local_noise_sigma": magnitude * scale if condition == "gaussian_noise" else 0.0,
            **metrics,
            "oracle_performance_gain": oracle["actual_performance_gain"],
            "oracle_minus_perturbed_gain": (
                oracle["actual_performance_gain"] - metrics["actual_performance_gain"]
            ),
        })
    return rows


def source_metadata() -> dict:
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip())
    return {
        "source_revision": revision, "source_dirty": dirty,
        "source_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
        "numpy_version": np.__version__, "python_version": platform.python_version(),
    }


def run_experiment(out_dir: Path, cfg: RunConfig, seeds: list[int], *, noise_root: int = 0) -> None:
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or noise_root < 0:
        raise ValueError("Training seeds must be distinct and nonnegative; noise root must be nonnegative")
    mdp = make_branching_mdp()
    metadata = {
        "experiment": "002-controlled-critic-error", "environment": "branching_mdp_v1",
        "run_config": asdict(cfg), "methods": ["qac_full"], "seeds": seeds,
        "noise_multipliers": NOISE_MULTIPLIERS, "noise_replicates": NOISE_REPLICATES,
        "additive_biases": ADDITIVE_BIASES, "noise_root": noise_root,
        "gap_scale": "max(RMS unordered pairwise action-value gaps at each state, 1e-6)",
        "gap_floor": GAP_FLOOR, "ranking_gap_tolerance": 1e-6,
        "gain_zero_tolerance": GAIN_ZERO_TOLERANCE,
        "checkpoint_timing": "after completed transition updates; state is the next decision state",
        **source_metadata(),
    }
    # Output files can make a checkout dirty without changing the implementation.
    identity = {key: value for key, value in metadata.items() if key != "source_dirty"}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    provenance = {"run_id": run_id, "source_revision": metadata["source_revision"]}
    metadata["run_id"] = run_id
    checkpoints, candidates, snapshots = [], [], []
    for seed in seeds:
        def capture(step, logits, critic, state, *, run_seed=seed):
            snapshots.append((run_seed, step, state, logits, critic))
            rows = checkpoint_candidates(mdp, logits, state, run_seed, step, cfg, noise_root=noise_root)
            candidates.extend({**provenance, **row} for row in rows)

        history, _ = train(mdp, "qac_full", seed, cfg, checkpoint_callback=capture)
        states = {step: state for run_seed, step, state, _, _ in snapshots if run_seed == seed}
        checkpoints.extend({**provenance, **row, "state": states[row["step"]]} for row in history)
        print(f"qac_full seed={seed} checkpoints={len(history)} J(end)={history[-1]['expected_return']:.6f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "checkpoints.csv", checkpoints)
    write_csv(out_dir / "counterfactual_updates.csv", candidates)
    np.savez_compressed(
        out_dir / "frozen_checkpoints.npz",
        seed=np.array([s[0] for s in snapshots]), step=np.array([s[1] for s in snapshots]),
        state=np.array([s[2] for s in snapshots]), logits=np.stack([s[3] for s in snapshots]),
        critic_q=np.stack([s[4] for s in snapshots]), run_id=np.array(run_id),
    )
    (out_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out_dir / "source_revision.txt").write_text(metadata["source_revision"] + "\n", encoding="utf-8")
    print(f"Records written to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--actor-lr", type=float, default=0.20)
    parser.add_argument("--critic-lr", type=float, default=0.12)
    parser.add_argument("--noise-root", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/002-controlled-critic-error/data"))
    args = parser.parse_args()
    run_experiment(args.out_dir, RunConfig(args.steps, args.eval_every, args.actor_lr, args.critic_lr),
                   args.seeds, noise_root=args.noise_root)


if __name__ == "__main__":
    main()
