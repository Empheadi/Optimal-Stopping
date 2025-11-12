"""Dataset generation for the wave-off optimal stopping problem."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .baseline import BaselineThresholdPolicy
from .env import ACTION_CONTINUE, ACTION_WAVE_OFF, WaveOffEnv


def generate_dataset(
    episodes: int,
    output: Path,
    seed: int = 0,
    policy: BaselineThresholdPolicy | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    env = WaveOffEnv(seed=rng.integers(1_000_000_000))
    policy = policy or BaselineThresholdPolicy()

    observations: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    costs: List[int] = []
    next_observations: List[np.ndarray] = []
    dones: List[bool] = []
    episode_ids: List[int] = []
    is_weights: List[float] = []
    randomizations: List[Dict[str, float]] = []
    outcomes: List[Dict[str, object]] = []

    for ep in range(episodes):
        obs = env.reset(seed=rng.integers(1_000_000_000))
        done = False
        while not done:
            action = int(policy(obs))
            if action not in (ACTION_CONTINUE, ACTION_WAVE_OFF):
                action = ACTION_CONTINUE
            next_obs, reward, cost, done, _ = env.step(action)

            observations.append(obs.astype(np.float32, copy=False))
            actions.append(action)
            rewards.append(float(reward))
            costs.append(int(cost))
            next_observations.append(next_obs.astype(np.float32, copy=False))
            dones.append(bool(done))
            episode_ids.append(ep)
            is_weights.append(1.0)

            obs = next_obs

        outcome = env.outcome
        randomizations.append(env.current_randomization)
        outcomes.append(
            {
                "episode": ep,
                "mode": outcome.mode if outcome else "unknown",
                "touchdown_error": outcome.touchdown_error if outcome else None,
                "stern_clearance": outcome.stern_clearance if outcome else None,
            }
        )

    dataset = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.int32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "episode_ids": np.asarray(episode_ids, dtype=np.int32),
        "is_weights": np.asarray(is_weights, dtype=np.float32),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)

    meta = {
        "episodes": episodes,
        "randomizations": randomizations,
        "outcomes": outcomes,
        "policy": asdict(policy),
        "seed": seed,
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote dataset to {output}")
    print(f"Metadata written to {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2048, help="Number of episodes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset.npz"),
        help="Output path for the dataset",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(args.episodes, args.output, args.seed)


if __name__ == "__main__":
    main()
