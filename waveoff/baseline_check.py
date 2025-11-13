"""Diagnostics for the handcrafted baseline wave-off policy."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .baseline import BaselineThresholdPolicy
from .env import ACTION_CONTINUE, ACTION_WAVE_OFF, WaveOffEnv


@dataclass
class StepLog:
    """Container capturing a single simulation step."""

    time: float
    distance_to_stern: float
    glide_slope_error: float
    deck_height: float
    aircraft_height: float
    action: int


@dataclass
class EpisodeLog:
    """Trajectory log plus metadata for an episode."""

    history: List[StepLog]
    outcome: Dict[str, object]
    reward: float


# ---------------------------------------------------------------------------
# Baseline policy checks
# ---------------------------------------------------------------------------
def sanity_check_policy(policy: BaselineThresholdPolicy) -> None:
    """Check that the baseline policy waves off when the error is large."""

    # Observation format: [error, error_rate, distance, vz, pitch, pitch_rate, heave, burble]
    near_stern = 30.0
    severe_error = np.array([-2.5, 0.0, near_stern, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert (
        policy(severe_error) == ACTION_WAVE_OFF
    ), "Policy should wave off for large negative errors."

    stable_error = np.array([0.0, 0.0, near_stern, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert (
        policy(stable_error) == ACTION_CONTINUE
    ), "Policy should continue when errors are acceptable."

    fast_closing = np.array([0.0, -1.0, near_stern, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert (
        policy(fast_closing) == ACTION_WAVE_OFF
    ), "Policy should wave off for excessive closing rates."


# ---------------------------------------------------------------------------
# Simulation utilities
# ---------------------------------------------------------------------------
def compute_altitudes(obs: np.ndarray, env: WaveOffEnv) -> Tuple[float, float]:
    """Return deck-relative and absolute aircraft heights for an observation."""

    e, _, s, _, pitch, _, heave, _ = obs
    deck_height = WaveOffEnv._deck_height(s, heave, pitch)
    glide_rad = math.radians(env.config.glide_slope_deg)
    reference_z = math.tan(glide_rad) * (s + env.config.s_offset)
    calm_deck = WaveOffEnv._deck_height(s, 0.0, 0.0)
    aircraft_height = e + deck_height + (reference_z - calm_deck)
    return deck_height, aircraft_height


def roll_out_episode(env: WaveOffEnv, policy: BaselineThresholdPolicy) -> EpisodeLog:
    """Run a single baseline episode and record its trajectory."""

    obs = env.reset()
    done = False
    reward = 0.0
    history: List[StepLog] = []
    time_elapsed = 0.0
    dt = env.config.dt

    while not done:
        deck_height, aircraft_height = compute_altitudes(obs, env)
        action = int(policy(obs))
        history.append(
            StepLog(
                time=time_elapsed,
                distance_to_stern=float(obs[2]),
                glide_slope_error=float(obs[0]),
                deck_height=deck_height,
                aircraft_height=aircraft_height,
                action=action,
            )
        )
        obs, reward, _, done, _ = env.step(action)
        time_elapsed += dt

    outcome = env.outcome
    outcome_dict = asdict(outcome) if outcome is not None else {"mode": "unknown"}
    return EpisodeLog(history=history, outcome=outcome_dict, reward=reward)


def summarise_trials(trials: List[EpisodeLog]) -> None:
    """Print a short summary over many baseline trials."""

    mode_counts: Dict[str, int] = {}
    rewards = []
    for log in trials:
        mode = log.outcome.get("mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        rewards.append(log.reward)

    print("Baseline outcomes:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode:>18}: {count}")
    if rewards:
        print(f"Average terminal reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def plot_landing(log: EpisodeLog, env: WaveOffEnv, output_path: str) -> None:
    """Save a landing performance plot for a single trajectory."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting dependency
        raise RuntimeError("matplotlib is required for plotting.") from exc

    x = [-entry.distance_to_stern for entry in log.history]
    aircraft = [entry.aircraft_height for entry in log.history]
    deck = [entry.deck_height for entry in log.history]

    plt.figure(figsize=(8, 4))
    plt.plot(x, aircraft, label="Aircraft", linewidth=2)
    plt.plot(x, deck, label="Deck", linewidth=2)
    plt.axvline(env.config.trap_distance, color="k", linestyle="--", label="Trap distance")
    plt.xlabel("Distance past stern (m)")
    plt.ylabel("Height (m)")
    plt.title(f"Baseline outcome: {log.outcome.get('mode', 'unknown')}")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved landing plot to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baseline wave-off diagnostics")
    parser.add_argument("--episodes", type=int, default=10, help="Number of baseline trials to simulate")
    parser.add_argument("--seed", type=int, default=0, help="Environment RNG seed")
    parser.add_argument("--plot", type=str, default="baseline_trial.png", help="Output path for the first episode plot")
    args = parser.parse_args(list(argv) if argv is not None else None)

    env = WaveOffEnv(seed=args.seed)
    policy = BaselineThresholdPolicy()

    sanity_check_policy(policy)
    trials: List[EpisodeLog] = []
    for _ in range(args.episodes):
        trials.append(roll_out_episode(env, policy))

    summarise_trials(trials)
    if trials:
        plot_landing(trials[0], env, args.plot)


if __name__ == "__main__":
    main()
