"""Dataset generation for the wave-off optimal stopping problem.

This module simulates episodes using a fixed behaviour policy (typically the
handcrafted baseline) and records (s, a, r, s') tuples for offline RL.

The resulting dataset is stored as a compressed .npz file with the following
arrays:

- observations:        shape (N, obs_dim)
- actions:             shape (N,)
- rewards:             shape (N,)
- costs:               shape (N,)
- next_observations:   shape (N, obs_dim)
- dones:               shape (N,)
- episode_ids:         shape (N,)

where episode_ids[i] is an integer identifier for the episode to which the i-th
transition belongs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tqdm.auto import trange

from baseline import BaselineThresholdPolicy
from env import ACTION_CONTINUE, ACTION_WAVE_OFF, WaveOffEnv


def generate_dataset(
    env: WaveOffEnv,
    policy: BaselineThresholdPolicy,
    episodes: int,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Roll out the behaviour policy to create an offline dataset.

    Parameters
    ----------
    env:
        Wave-off environment instance. This function will call `reset()` before
        each episode and will not modify `env.config`.
    policy:
        Behaviour policy to use for data collection (typically the handcrafted
        baseline).
    episodes:
        Number of episodes to simulate.
    seed:
        Optional random seed to re-seed the environment RNG before generation.

    Returns
    -------
    dataset:
        A dictionary of numpy arrays ready to be saved to disk.
    """

    # Global seeding (if the environment exposes a seed method)
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)

    observations: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    costs: List[float] = []
    next_observations: List[np.ndarray] = []
    dones: List[float] = []
    episode_ids: List[int] = []

    cfg = env.config
    ep_id = 0

    # tqdm progress bar over episodes
    for ep in trange(episodes, desc="Generating episodes"):
        obs = env.reset()
        done = False
        wave_off_locked = False  # local latch mirroring env's wave-off mode

        while not done:
            # Current distance to stern
            s = float(obs[2])
            wave_off_allowed = cfg.wave_off_s_min <= s <= cfg.wave_off_s_max

            # 1) Behaviour policy proposes an action
            proposed_action = int(policy(obs))
            if proposed_action not in (ACTION_CONTINUE, ACTION_WAVE_OFF):
                proposed_action = ACTION_CONTINUE

            # 2) Apply window + latching consistently with env semantics
            if wave_off_locked:
                # Once latched, the effective action is always WAVE_OFF
                action = ACTION_WAVE_OFF
            else:
                if proposed_action == ACTION_WAVE_OFF and not wave_off_allowed:
                    # Wave-off requested outside the allowed window -> ignore
                    action = ACTION_CONTINUE
                else:
                    action = proposed_action

                # First valid wave-off within the allowed window -> latch
                if action == ACTION_WAVE_OFF and wave_off_allowed:
                    wave_off_locked = True

            # 3) Step environment with the effective action
            next_obs, reward, cost, done, _info = env.step(action)

            # 4) Log transition (effective action, consistent with env dynamics)
            observations.append(obs.astype(np.float32))
            actions.append(action)
            rewards.append(float(reward))
            costs.append(float(cost))
            next_observations.append(next_obs.astype(np.float32))
            dones.append(float(done))
            episode_ids.append(ep_id)

            obs = next_obs

        ep_id += 1

    # Stack everything into contiguous arrays.
    dataset = dict(
        observations=np.stack(observations, axis=0),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        costs=np.asarray(costs, dtype=np.float32),
        next_observations=np.stack(next_observations, axis=0),
        dones=np.asarray(dones, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )
    return dataset


def save_dataset(dataset: Dict[str, np.ndarray], path: Path) -> None:
    """Save the dataset as a compressed .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    print(f"Saved dataset to {path} (size={path.stat().st_size / 1e6:.2f} MB)")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate offline dataset for CQL.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100000,
        help="Number of episodes to simulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for environment and behaviour policy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/waveoff_dataset.npz"),
        help="Path to save the generated dataset (.npz).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    env = WaveOffEnv(seed=args.seed)
    policy = BaselineThresholdPolicy()

    dataset = generate_dataset(env, policy, args.episodes, seed=args.seed)
    save_dataset(dataset, args.output)


if __name__ == "__main__":
    main()
