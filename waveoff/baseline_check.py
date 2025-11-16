"""Diagnostics for the handcrafted baseline wave-off policy."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

from baseline import BaselineThresholdPolicy
from env import ACTION_CONTINUE, ACTION_WAVE_OFF, WaveOffEnv
import matplotlib
import os

# Try a GUI backend. TkAgg is usually available on Windows.
matplotlib.use("TkAgg")


@dataclass
class StepLog:
    """Container capturing a single simulation step and policy checks."""

    time: float
    distance_to_stern: float
    glide_slope_error: float
    error_rate: float
    dynamic_error_threshold: float
    rate_threshold: float
    deck_height: float
    aircraft_height: float
    action: int
    wave_off_reason: Optional[str]  # "error", "rate", or None
    step_reward: float              # reward received at this step
    cum_reward: float               # accumulated reward up to and including this step


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
    """Check that the baseline policy responds sensibly in simple scenarios.

    We verify:
    - Large negative error near the stern -> wave-off.
    - Small error and zero closing rate far from the stern -> continue.
    - Zero error but very negative closing rate near the stern -> wave-off.
    """

    # Observation format: [error, error_rate, distance, vz, pitch, pitch_rate, heave, burble]
    near_stern = 30.0
    far_from_stern = 200.0  # > activation_distance so dynamic threshold = error_threshold

    # 1) Large negative error near the stern -> must wave off
    severe_error = np.array(
        [-2.5, 0.0, near_stern, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    assert (
        policy(severe_error) == ACTION_WAVE_OFF
    ), "Policy should wave off for large negative errors near stern."

    # 2) Zero error and zero rate far from the stern -> should continue
    stable_error = np.array(
        [0.0, 0.0, far_from_stern, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    assert (
        policy(stable_error) == ACTION_CONTINUE
    ), "Policy should continue when far from the stern with acceptable errors."

    # 3) Zero error but very negative closing rate near the stern -> must wave off
    fast_closing = np.array(
        [0.0, -1.0, near_stern, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    assert (
        policy(fast_closing) == ACTION_WAVE_OFF
    ), "Policy should wave off for excessive closing rates near stern."



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


def _compute_policy_thresholds(
    obs: np.ndarray, policy: BaselineThresholdPolicy
) -> Tuple[float, float, Optional[str]]:
    """Recompute the policy's internal thresholds and the triggering reason.

    Returns
    -------
    dynamic_threshold:
        The distance-dependent error threshold used for the glide-slope error.
    rate_threshold:
        The fixed threshold used for the closing rate.
    reason:
        "error" if the error condition triggered the wave-off,
        "rate" if the rate condition triggered the wave-off,
        or None if the policy chose to continue.
    """

    e = float(obs[0])
    e_rate = float(obs[1])
    s = float(obs[2])

    dynamic_threshold = policy.error_threshold + policy.closing_bias * np.sqrt(
        max(0.0, policy.activation_distance - s)
    )
    reason: Optional[str] = None
    if e < dynamic_threshold:
        reason = "error"
    elif e_rate < policy.rate_threshold:
        reason = "rate"
    return float(dynamic_threshold), float(policy.rate_threshold), reason


def roll_out_episode(env: WaveOffEnv, policy: BaselineThresholdPolicy) -> EpisodeLog:
    """Run a single baseline episode and record its trajectory and checks.

    Once a valid wave-off (within the allowed s window) is issued, we latch that decision:
    - The agent is no longer queried.
    - We keep sending ACTION_WAVE_OFF to the environment until done.
    """

    obs = env.reset()
    done = False
    reward_total = 0.0
    history: List[StepLog] = []
    time_elapsed = 0.0
    dt = env.config.dt

    cfg = env.config
    wave_off_locked = False  # has a (valid) wave-off been triggered already?

    while not done:
        deck_height, aircraft_height = compute_altitudes(obs, env)
        s = float(obs[2])

        # Is wave-off allowed at the *current* distance?
        wave_off_allowed = cfg.wave_off_s_min <= s <= cfg.wave_off_s_max

        if not wave_off_locked:
            # Normal decision phase: compute thresholds and let the policy decide.
            dynamic_thresh, rate_thresh, reason = _compute_policy_thresholds(obs, policy)
            policy_action = int(policy(obs))

            # Sanity check: recomputed rule must match the policy's output.
            expected_action = ACTION_CONTINUE
            if reason is not None:
                expected_action = ACTION_WAVE_OFF
            if policy_action != expected_action:
                print(
                    f"[WARN] Policy action mismatch at t={time_elapsed:.2f}s: "
                    f"expected {expected_action}, got {policy_action}"
                )

            # Apply the wave-off window: outside the window, env will ignore wave-off.
            if policy_action == ACTION_WAVE_OFF and not wave_off_allowed:
                # Env will treat this as CONTINUE.
                action = ACTION_CONTINUE
                # Mark why the wave-off did not take effect.
                wave_off_reason = "window_blocked"
            else:
                action = policy_action
                wave_off_reason = reason if action == ACTION_WAVE_OFF else None

            # Latch only if a wave-off is BOTH commanded and allowed.
            if action == ACTION_WAVE_OFF and wave_off_allowed:
                wave_off_locked = True

        else:
            # Wave-off has already been triggered earlier (within window).
            # For realism and consistency with env, we force the action
            # to remain WAVE_OFF; the policy is no longer consulted.
            dynamic_thresh, rate_thresh, _ = _compute_policy_thresholds(obs, policy)
            policy_action = ACTION_WAVE_OFF
            action = ACTION_WAVE_OFF
            wave_off_reason = "latched"

        # Environment sees the *effective* action (after window + latching),
        # exactly as in env.py.
        next_obs, reward, _, done, _ = env.step(action)
        time_elapsed += dt
        reward_total += reward

        # Log AFTER we know the reward and updated cumulative reward
        history.append(
            StepLog(
                time=time_elapsed,  # time *after* this step
                distance_to_stern=float(next_obs[2]),
                glide_slope_error=float(next_obs[0]),
                error_rate=float(next_obs[1]),
                dynamic_error_threshold=dynamic_thresh,
                rate_threshold=rate_thresh,
                deck_height=deck_height,
                aircraft_height=aircraft_height,
                action=action,
                wave_off_reason=reason if action == ACTION_WAVE_OFF else None,
                step_reward=float(reward),
                cum_reward=float(reward_total),
            )
        )

        obs = next_obs

    outcome = env.outcome
    outcome_dict = asdict(outcome) if outcome is not None else {"mode": "unknown"}
    return EpisodeLog(history=history, outcome=outcome_dict, reward=reward_total)


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


def print_decision_trace(log: EpisodeLog) -> None:
    """Print a detailed step-by-step trace for one episode.

    This exposes, for each decision point:
    - distance to stern,
    - glide-slope error and its dynamic threshold,
    - error rate and its threshold,
    - chosen action and which condition (if any) triggered the wave-off.
    """

    header = (
        "idx   t[s]   s[m]     e[m]    e_thr[m]   e_dot[m/s]  rate_thr  "
        "action  reason   r_step   R_cum"
    )

    print("\nDetailed decision trace for one episode:")
    print(header)
    print("-" * len(header))

    for idx, step in enumerate(log.history):
        action_str = "CONT" if step.action == ACTION_CONTINUE else "WAVE"
        reason = step.wave_off_reason or "-"
        print(
            f"{idx:3d}  {step.time:5.2f}  {step.distance_to_stern:6.1f}  "
            f"{step.glide_slope_error:7.3f}  {step.dynamic_error_threshold:9.3f}  "
            f"{step.error_rate:10.3f}  {step.rate_threshold:8.3f}  "
            f"{action_str:>6}  {reason:>7}  "
            f"{step.step_reward:7.3f}  {step.cum_reward:7.3f}"
        )

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def plot_landing(log: EpisodeLog, env: WaveOffEnv, output_path: Optional[str] = None) -> None:
    """Show (and optionally save) diagnostic plots for a single trajectory.

    The figure contains:
    - Vertical geometry: aircraft vs deck height.
    - Glide-slope error vs dynamic error threshold.
    - Error rate vs rate threshold.

    All panels share the same horizontal axis (distance past the stern).
    """

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting dependency
        raise RuntimeError("matplotlib is required for plotting.") from exc

    # Horizontal axis: distance past stern (m), so 0 at stern, positive over deck.
    x = [-entry.distance_to_stern for entry in log.history]

    aircraft = [entry.aircraft_height for entry in log.history]
    deck = [entry.deck_height for entry in log.history]

    errors = [entry.glide_slope_error for entry in log.history]
    error_thresholds = [entry.dynamic_error_threshold for entry in log.history]

    rates = [entry.error_rate for entry in log.history]
    rate_thresholds = [entry.rate_threshold for entry in log.history]

    actions = [entry.action for entry in log.history]

    # Identify the first wave-off decision, if any.
    wave_index = next((i for i, a in enumerate(actions) if a == ACTION_WAVE_OFF), None)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))

    # Panel 1: geometry
    ax0 = axes[0]
    ax0.plot(x, aircraft, label="Aircraft", linewidth=2)
    ax0.plot(x, deck, label="Deck", linewidth=2)
    ax0.axvline(
        env.config.trap_distance,
        linestyle="--",
        linewidth=1,
        color="k",
        label="Trap distance",
    )
    if wave_index is not None:
        ax0.axvline(
            x[wave_index],
            linestyle=":",
            linewidth=1.5,
            color="r",
            label="Wave-off command",
        )
    ax0.set_ylabel("Height (m)")
    ax0.set_title(f"Baseline outcome: {log.outcome.get('mode', 'unknown')}")
    ax0.grid(True, linestyle=":")
    ax0.legend(loc="best")

    # Panel 2: glide-slope error vs threshold
    ax1 = axes[1]
    ax1.plot(x, errors, label="Glide-slope error e")
    ax1.plot(x, error_thresholds, label="Dynamic error threshold e_thr", linestyle="--")
    if wave_index is not None:
        # Highlight the point where the wave-off condition first triggered.
        ax1.axvline(
            x[wave_index],
            linestyle=":",
            linewidth=1.5,
            color="r",
            label="Wave-off command",
        )
    ax1.set_ylabel("Error (m)")
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="best")

    # Panel 3: error rate vs threshold
    ax2 = axes[2]
    ax2.plot(x, rates, label="Error rate e_dot")
    # rate_threshold is constant, just take the first
    if rate_thresholds:
        ax2.axhline(rate_thresholds[0], label="Rate threshold", linestyle="--")
    if wave_index is not None:
        ax2.axvline(
            x[wave_index],
            linestyle=":",
            linewidth=1.5,
            color="r",
            label="Wave-off command",
        )
    ax2.set_xlabel("Distance past stern (m)")
    ax2.set_ylabel("Error rate (m/s)")
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="best")

    plt.tight_layout()

    # Optional save
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved landing plot to {output_path}")

    # Always show the figure interactively (PyCharm will open a window if backend supports it)
    plt.show()



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baseline wave-off diagnostics")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of baseline trials to simulate",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Environment RNG seed"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="baseline_trial.png",
        help=(
            "Base output path for episode plots. "
            "For multiple episodes, suffix _ep{idx} will be appended."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    env = WaveOffEnv(seed=args.seed)
    policy = BaselineThresholdPolicy()

    # Optional: enable if you want the assertions
    # sanity_check_policy(policy)

    trials: List[EpisodeLog] = []
    for _ in range(args.episodes):
        trials.append(roll_out_episode(env, policy))

    summarise_trials(trials)

    if not trials:
        return

    # Print detailed trace and plot for EVERY episode
    base_plot = args.plot or ""  # allow "" to mean "no saving"
    root, ext = os.path.splitext(base_plot) if base_plot else ("", "")

    for idx, log in enumerate(trials):
        print(f"\n===== Episode {idx} =====")
        print_decision_trace(log)

        if base_plot:
            # e.g. baseline_trial_ep0.png, baseline_trial_ep1.png, ...
            ep_path = f"{root}_ep{idx}{ext}"
        else:
            ep_path = None  # no saving, just show

        plot_landing(log, env, ep_path)



if __name__ == "__main__":
    main()
