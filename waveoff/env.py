"""Environment model for the carrier wave-off optimal stopping problem.

This module defines a small, self-contained environment that mimics the
decision a pilot must make when approaching an aircraft carrier:
- CONTINUE the landing, or
- WAVE OFF (go around) to avoid a dangerous landing.

The environment is designed for offline / batch RL:
- The *state* represents glide-slope tracking error, relative deck motion,
  and a simplified "burble" (air disturbance) model.
- The *action* is a binary choice: continue vs wave-off.
- The *reward* encourages safe landings and safe wave-offs, and heavily
  penalises ramp strikes or unsafe wave-offs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np


# Public action constants used by the rest of the code base.  These are
# intentionally small integers so they can be used as indices into Q-values.
ACTION_CONTINUE = 0
ACTION_WAVE_OFF = 1


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WaveOffProfile:
    """Parameters describing the fixed wave-off climb profile.

    Once a wave-off is commanded, the aircraft attempts to follow a simple
    vertical climb profile:
    - It asymptotically approaches ``commanded_climb_rate`` (m/s),
    - With a first-order engine time constant ``engine_time_constant``,
    - And a hard cap on vertical acceleration ``max_vertical_accel``.

    These parameters only matter *after* a wave-off is triggered.
    """

    commanded_climb_rate: float = 8.0  # Target vertical climb speed (m/s)
    engine_time_constant: float = 1.0  # How quickly thrust responds (s)
    max_vertical_accel: float = 3.0  # Cap on vertical acceleration (m/s^2)


@dataclass
class NoiseParams:
    """Noise parameters for glide-slope error dynamics.

    We model glide-slope tracking error with a crude AR process and inject
    Gaussian noise into:
    - ``eta_std``: noise on the error itself,
    - ``nu_std``: noise on the error rate.

    This lets the environment capture random, unmodelled disturbances.
    """

    # eta_std: float = 0.2
    # nu_std: float = 0.05
    eta_std: float = 1
    nu_std: float = 0.05


@dataclass
class EnvironmentConfig:
    """Configuration for the wave-off optimal stopping environment.

    Most of these parameters are physics or geometry constants that define
    what “normal” looks like for a carrier approach. Others control the
    stochastic error model and the reward structure.
    """

    # Simulation + RL parameters
    dt: float = 0.1                      # Simulation time step (seconds)
    gamma: float = 0.99                  # Discount factor used downstream

    # Nominal approach geometry / kinematics
    glide_slope_deg: float = 3.5         # Glide-slope angle from horizon
    approach_speed: float = 65.0         # Along-deck speed (m/s ≈ 126 knots)
    s_init: float = 1200.0               # Initial distance to stern (m)
    s_offset: float = 50.0               # Offset so ref height > 0 at start

    # Where and how we judge the landing
    trap_distance: float = 75.0          # Distance past stern at which we
                                         # evaluate landing performance (m)
    target_wire: float = 53.6            # Ideal touchdown distance (m)
    landing_tolerance: float = 12.0      # Scale for landing reward shaping (m)

    # Safety clearances
    ramp_clearance: float = 3.0          # Minimum ramp clearance (m)
    stern_clearance: float = 6.0         # Clearance needed during wave-off (m)

    # Wave-off success reward shaping
    wave_off_optimal_clearance: float = 10.0  # "Ideal" stern clearance after wave-off (m)
    wave_off_success_base_reward: float = 0.2  # Baseline reward for any successful wave-off
    wave_off_success_bonus: float = 0.8  # Extra bonus at the ideal clearance


    # Wave-off decision window in s (distance to stern, in metres)
    # Wave-off commands outside this window are ignored (treated as CONTINUE).
    wave_off_s_min: float = 100.0  # Inner limit (closer to stern)
    wave_off_s_max: float = 600.0  # Outer limit (farther from stern)

    # Safety / numerical guardrails
    max_time: float = 80.0               # Hard episode cutoff (seconds)

    # Glide-slope error dynamics (AR-like model)
    phi: float = 0.99                    # AR coefficient for error
    rho: float = 0.85                    # AR coefficient for error rate
    psi: float = 0.05                    # Coupling between rate and error

    # Coupling from disturbances into the error
    beta_burble: float = 0.3             # Coupling from burble disturbance
    beta_pitch: float = 0.5              # Coupling from deck pitch rate

    # Initial condition statistics
    initial_error_mean: float = -5
    initial_error_std: float = 10
    initial_rate_std: float = 0.4

    # Nominal deck motion (before randomisation)
    heave_amplitude: float = 0.2         # Deck heave amplitude (m)
    heave_frequency: float = 0.5         # Heave frequency (Hz)
    pitch_amplitude: float = math.radians(1.5)  # Pitch amplitude (rad)
    pitch_frequency: float = 0.8         # Pitch frequency (Hz)

    # Nominal burble model (Gaussian bump in along-deck space)
    burble_strength: float = 2.0
    burble_mean: float = 40.0
    burble_std: float = 25.0

    # Embedded sub-configs
    wave_off_profile: WaveOffProfile = WaveOffProfile()
    noise: NoiseParams = NoiseParams()


@dataclass
class RandomizationParams:
    """Randomization ranges for domain randomization.

    At the start of each episode, we sample a value for each parameter in
    these ranges (uniformly). This supports domain randomisation:
    every episode can have different sea state, burble shape, and approach
    speed, which helps stress-test learned policies.
    """

    heave_amplitude: Tuple[float, float] = (0.0, 0.6)
    heave_frequency: Tuple[float, float] = (0.2, 0.8)
    pitch_amplitude: Tuple[float, float] = (math.radians(0.5), math.radians(2.0))
    pitch_frequency: Tuple[float, float] = (0.4, 1.2)
    burble_strength: Tuple[float, float] = (1.0, 4.5)
    burble_mean: Tuple[float, float] = (25.0, 60.0)
    burble_std: Tuple[float, float] = (15.0, 45.0)
    approach_speed: Tuple[float, float] = (62.0, 68.0)
    eta_std: Tuple[float, float] = (0.15, 0.35)
    nu_std: Tuple[float, float] = (0.04, 0.09)


@dataclass
class EpisodeOutcome:
    """Information captured at the end of an episode.

    Attributes
    ----------
    mode:
        High-level label for how the episode ended, e.g.
        ``"landed"``, ``"wave_off_success"``, ``"ramp_strike"``, etc.
    touchdown_error:
        Longitudinal error at touchdown relative to the target wire, in metres.
        Only defined when ``mode == "landed"``.
    stern_clearance:
        Minimum vertical clearance at the stern when crossing it, if relevant.
    randomization:
        The specific randomised parameters used for this episode.
    """

    mode: str
    touchdown_error: Optional[float]
    stern_clearance: Optional[float]
    randomization: Dict[str, float]


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------


class WaveOffEnv:
    """Wave-off decision environment with simplified glide-path dynamics.

    This is intentionally minimal but still physically interpretable. It is
    *not* a full flight dynamics simulator; it just captures the key factors
    for the wave-off vs continue decision.

    Observation vector (8D)
    -----------------------
    ``obs = [e, e_dot, s, v_z, pitch, pitch_rate, heave, burble]``

    - ``e``:      Glide-slope error (m). Positive => above reference path.
    - ``e_dot``:  Time derivative of error (m/s).
    - ``s``:      Distance from the carrier stern along the deck (m).
                  s > 0: still before stern, s = 0: over stern,
                  s < 0: past stern towards trap point.
    - ``v_z``:    Vertical speed of the aircraft (m/s).
                  Negative ≈ descending on the glide slope.
    - ``pitch``:  Deck pitch angle (rad).
    - ``pitch_rate``: Time derivative of deck pitch (rad/s).
    - ``heave``:  Vertical displacement of the deck (m).
    - ``burble``: Scalar representing the burble disturbance at the current s.

    Action space
    ------------
    - ``ACTION_CONTINUE`` (0): keep the approach and stay on glide slope.
    - ``ACTION_WAVE_OFF`` (1): command a go-around / wave-off.

    Step return
    -----------
    ``obs, reward, cost, done, info`` where:

    - ``reward``: shaped reward, large negative on failure, positive for
      good landings, small positive for safe wave-offs.
    - ``cost``:   binary indicator (0/1) for “catastrophic” failure events.
    - ``done``:   whether the episode has terminated.
    - ``info``:   currently used to expose ``"stern_clearance"`` when
      relevant.
    """

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        randomization: Optional[RandomizationParams] = None,
        seed: Optional[int] = None,
    ) -> None:
        # Save configuration + randomisation ranges
        self.config = config or EnvironmentConfig()
        self.randomization = randomization or RandomizationParams()

        # Numpy RNG controlling all stochasticity inside the environment
        self.rng = np.random.default_rng(seed)

        # Internal state (8D observation vector) or None before reset()
        self._state: Optional[np.ndarray] = None

        # Simulation bookkeeping
        self._time = 0.0
        self._waved_off = False
        self._wave_off_vz = 0.0  # Vertical speed if we commit to wave-off

        # Per-episode diagnostics
        self._outcome: Optional[EpisodeOutcome] = None

        # Concrete values sampled from RandomizationParams for this episode
        self._current_params: Dict[str, float] = {}

        # NEW: one-time evaluation latches
        self._stern_checked = False  # compute stern clearance exactly once
        self._stern_clearance_value = None  # store that one-time clearance
        self._touched_down = False  # latch touchdown when it first happens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed(self, seed: Optional[int]) -> None:
        """Set the seed for the internal RNG.

        This is a simple wrapper used by dataset generation / experiments
        to make the environment's stochasticity reproducible.
        """
        self.rng = np.random.default_rng(seed)

    def reset(
        self,
        seed: Optional[int] = None,
        randomization: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Reset the environment and start a new episode.

        Parameters
        ----------
        seed:
            Optional seed to re-seed the internal RNG for this episode.
        randomization:
            Optional dictionary that overrides any of the sampled
            domain-randomised parameters.

        Returns
        -------
        obs:
            The initial 8D observation as a NumPy array (dtype float32).
        """

        self._time = 0.0
        self._waved_off = False
        self._stern_checked = False
        self._stern_clearance_value = None
        self._touched_down = False

        if seed is not None:
            # Allow per-episode seeding, useful for deterministic test cases.
            self.rng = np.random.default_rng(seed)

        cfg = self.config

        # Sample this episode's randomised parameters (heave, pitch, burble, etc.)
        # and cache them so step() uses consistent values.
        params = self._sample_randomization(randomization)
        self._current_params = params

        # --- Initial glide-slope state ---------------------------------
        # Initial error around a mean positive bias: we start slightly high.
        e0 = self.rng.normal(cfg.initial_error_mean, cfg.initial_error_std)

        # Initial rate-of-change of error (m/s).
        e_dot0 = self.rng.normal(0.0, cfg.initial_rate_std)

        # Initial along-deck distance to the stern.
        s0 = cfg.s_init

        # Randomise phases for sine-wave deck motion so deck heave/pitch do
        # not repeat exactly across episodes.
        pitch_phase = self.rng.uniform(0, 2 * math.pi)
        heave_phase = self.rng.uniform(0, 2 * math.pi)
        self._phases = {"pitch": pitch_phase, "heave": heave_phase}

        # Reset episode-level bookkeeping
        self._time = 0.0
        self._waved_off = False

        # Vertical speed implied by flying down the glide slope
        approach_speed = params.get("approach_speed", cfg.approach_speed)
        self._wave_off_vz = -approach_speed * math.sin(
            math.radians(cfg.glide_slope_deg)
        )

        # Evaluate deck motion and disturbance at t = 0
        pitch, pitch_rate = self._compute_pitch(0.0)
        heave, _ = self._compute_heave(0.0)
        burble = self._burble_effect(s0)

        # Assemble the initial observation vector.
        obs = np.array(
            [
                e0,
                e_dot0,
                s0,
                self._wave_off_vz,
                pitch,
                pitch_rate,
                heave,
                burble,
            ],
            dtype=np.float32,
        )

        self._state = obs.copy()
        self._outcome = None
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, int, bool, Dict[str, float]]:
        """Advance the simulation by one step.

        This function performs one synchronous update of:
        - the along-deck position,
        - the glide-slope tracking error and its rate,
        - the deck motion and burble disturbance,
        - the aircraft vertical speed (including wave-off dynamics),
        and then computes reward, cost, and termination conditions.
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if action not in (ACTION_CONTINUE, ACTION_WAVE_OFF):
            raise ValueError(f"Invalid action: {action}")

        cfg = self.config
        params = self._current_params

        # Unpack the previous observation.
        # We ignore pitch / heave here and recompute them from time + phases.
        e, e_dot, s, v_z, _, _, _, _ = self._state
        dt = cfg.dt

        # ------------------------------------------------------------------
        # 1) Advance along-deck position
        # ------------------------------------------------------------------
        v_app = params.get("approach_speed", cfg.approach_speed)
        s_next = s - v_app * dt
        self._time += dt

        # ------------------------------------------------------------------
        # 2) Apply wave-off window and register wave-off
        # ------------------------------------------------------------------
        # Only allow wave-off commands when s is within [wave_off_s_min, wave_off_s_max].
        wave_off_allowed = (cfg.wave_off_s_min <= s <= cfg.wave_off_s_max)

        # Effective action seen by the dynamics.
        if action == ACTION_WAVE_OFF and not wave_off_allowed:
            # Ignore wave-off outside the allowed window: treat as CONTINUE.
            effective_action = ACTION_CONTINUE
        else:
            effective_action = action

        # Once we wave off *within* the allowed window, we stay in wave-off mode.
        if effective_action == ACTION_WAVE_OFF and not self._waved_off:
            self._waved_off = True

        # ------------------------------------------------------------------
        # 3) Glide-slope error dynamics, deck motion, and vertical kinematics
        # ------------------------------------------------------------------
        # Stochastic error model (same parameters as before)
        eta = self.rng.normal(0.0, params.get("eta_std", cfg.noise.eta_std))
        nu = self.rng.normal(0.0, params.get("nu_std", cfg.noise.nu_std))

        # Disturbances and deck motion at the *current* time
        burble = self._burble_effect(s)
        pitch, pitch_rate = self._compute_pitch(self._time)
        heave, _ = self._compute_heave(self._time)

        # Start from the original AR-like update for the error dynamics
        e_dot_next = cfg.rho * e_dot + nu
        e_next = (
                cfg.phi * e
                + cfg.psi * e_dot * dt
                + cfg.beta_burble * burble * dt
                + eta
        )

        # ------------------------------------------------------------------
        # 4) Reference glide path (inertial frame) and current altitude
        # ------------------------------------------------------------------
        glide_rad = math.radians(cfg.glide_slope_deg)

        # Reference altitude at the current and next along-deck positions
        z_ref_curr = math.tan(glide_rad) * (s + cfg.s_offset)
        z_ref_next = math.tan(glide_rad) * (s_next + cfg.s_offset)

        # Aircraft absolute altitude at the *current* step (no deck motion)
        z_curr = z_ref_curr + e

        # ------------------------------------------------------------------
        # 5) Vertical speed update and wave-off profile
        # ------------------------------------------------------------------
        wave_profile = cfg.wave_off_profile
        if self._waved_off:
            # First-order engine response towards a commanded climb rate
            accel_cmd = (wave_profile.commanded_climb_rate - v_z) / max(
                wave_profile.engine_time_constant, 1e-6
            )
            # Symmetric cap on vertical acceleration
            accel = max(
                -wave_profile.max_vertical_accel,
                min(wave_profile.max_vertical_accel, accel_cmd),
            )
            v_z_next = v_z + accel * dt
        else:
            # Nominal vertical speed following the fixed glide slope
            v_z_next = -v_app * math.sin(glide_rad)

        # If we are in wave-off mode, actually climb using v_z_next and
        # recompute the glide-slope error from geometry.
        if self._waved_off:
            # Integrate vertical speed to get the next altitude
            z_next = z_curr + v_z_next * dt
            # New geometric error relative to the reference glide path at s_next
            e_next = z_next - z_ref_next
            # Consistent error rate from finite difference
            e_dot_next = (e_next - e) / dt
        else:
            # Original behaviour in the normal landing phase
            z_next = z_ref_next + e_next

        # Deck geometry (used ONLY for clearances / touchdown decisions)
        deck_at_s_next = self._deck_height(s_next, heave, pitch)
        deck_at_stern = self._deck_height(0.0, heave, pitch)

        next_burble = self._burble_effect(s_next)

        obs = np.array(
            [e_next, e_dot_next, s_next, v_z_next, pitch, pitch_rate, heave, next_burble],
            dtype=np.float32,
        )


        # ------------------------------------------------------------------
        # 6) Rewards, one-time stern clearance, touchdown detection, and termination
        # ------------------------------------------------------------------
        reward = 0.01  # basic per-step reward (your “otherwise just give basic reward”)
        cost = 0
        done = False
        info: Dict[str, float] = {}

        # (a) One-time stern clearance computation (first crossing of the stern)
        if not self._stern_checked and s_next <= 0.0:
            self._stern_checked = True
            self._stern_clearance_value = float(z_next - deck_at_stern)
            info["stern_clearance"] = self._stern_clearance_value

            # If NOT waved off and clearance is below ramp clearance at stern, it's a ramp strike
            if not self._waved_off and self._stern_clearance_value < cfg.ramp_clearance:
                reward = -100.0
                cost = 1
                done = True
                self._outcome = EpisodeOutcome(
                    mode="ramp_strike",
                    touchdown_error=None,
                    stern_clearance=self._stern_clearance_value,
                    randomization=params,
                )

        # (b) Touchdown detection & one-time touchdown reward (only over the deck)
        # Touchdown happens when aircraft intersects the physical deck (first time z <= deck)
        if not done and not self._touched_down and s_next <= 0.0 and z_next <= deck_at_s_next:
            self._touched_down = True
            # Compute where we touched relative to stern
            touchdown_x = -s_next  # distance past stern (m), positive on deck
            touchdown_error = touchdown_x - cfg.target_wire

            # One-time touchdown reward in [0, 1]
            reward = 1.0 - min(1.0, abs(touchdown_error) / cfg.landing_tolerance)
            reward *= 100
            done = True
            self._outcome = EpisodeOutcome(
                mode="landed",
                touchdown_error=float(touchdown_error),
                stern_clearance=self._stern_clearance_value,
                randomization=params,
            )

        # (c) Wave-off evaluation happens ONCE at stern crossing, using that one-time clearance
        if not done and self._waved_off and self._stern_checked:
            clearance = self._stern_clearance_value
            # Only decide at first stern crossing (the moment we latched it)
            # Success if clearance >= cfg.stern_clearance
            if clearance is not None:
                if clearance >= cfg.stern_clearance:
                    # --- Shaped reward for successful wave-off ---
                    # Idea:
                    # - reward peaks at some "ideal" clearance (wave_off_optimal_clearance)
                    # - reward decreases again for very large clearance
                    # - reward never goes to 0 (has a positive base)
                    base = cfg.wave_off_success_base_reward
                    bonus = cfg.wave_off_success_bonus
                    c_min = cfg.stern_clearance
                    c_opt = cfg.wave_off_optimal_clearance

                    # Width of the "good" region between minimum and optimal clearance.
                    # This sets how sharply the reward drops away from the optimum.
                    sigma = max(1e-3, 0.5 * max(0.0, c_opt - c_min))

                    # Smooth "bump" around c_opt:
                    # - shape(c_opt) = 1  -> reward = base + bonus (maximum)
                    # - shape falls towards 0 as |clearance - c_opt| grows
                    # So as clearance increases beyond c_opt, the reward decreases,
                    # but never below 'base'.
                    shape = math.exp(-0.5 * ((clearance - c_opt) / sigma) ** 2)

                    reward = base + bonus * shape
                    mode = "wave_off_success"
                else:
                    reward = -100.0
                    cost = 1
                    mode = "wave_off_failure"

                done = True
                self._outcome = EpisodeOutcome(
                    mode=mode,
                    touchdown_error=None,
                    stern_clearance=clearance,
                    randomization=params,
                )

        # (d) Emergency timeout (unchanged semantics)
        if not done and self._time >= cfg.max_time:
            done = True
            reward = -5.0
            self._outcome = EpisodeOutcome(
                mode="timeout",
                touchdown_error=None,
                stern_clearance=self._stern_clearance_value,
                randomization=params,
            )

        # Save state and return
        self._state = obs.copy()
        return obs, float(reward), cost, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_randomization(
        self, overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Sample a concrete set of randomisation parameters for one episode.

        Parameters
        ----------
        overrides:
            Optional dictionary for forcing specific values (useful for
            debugging / evaluation). Any keys present in ``overrides`` win
            over the sampled ones.

        Returns
        -------
        params:
            Dictionary mapping parameter names to sampled floats.
        """
        params: Dict[str, float] = {}
        ranges = asdict(self.randomization)
        for name, (low, high) in ranges.items():
            params[name] = self.rng.uniform(low, high)
        if overrides:
            params.update(overrides)
        return params

    def _compute_pitch(self, t: float) -> Tuple[float, float]:
        """Compute deck pitch angle and rate at time ``t``.

        Uses a simple sinusoid with amplitude/frequency sampled from
        domain-randomised parameters.
        """
        params = self._current_params
        amp = params.get("pitch_amplitude", self.config.pitch_amplitude)
        freq = params.get("pitch_frequency", self.config.pitch_frequency)
        phase = self._phases["pitch"]
        pitch = amp * math.sin(2.0 * math.pi * freq * t + phase)
        pitch_rate = amp * 2.0 * math.pi * freq * math.cos(
            2.0 * math.pi * freq * t + phase
        )
        return pitch, pitch_rate

    def _compute_heave(self, t: float) -> Tuple[float, float]:
        """Compute deck heave displacement and rate at time ``t``."""
        params = self._current_params
        amp = params.get("heave_amplitude", self.config.heave_amplitude)
        freq = params.get("heave_frequency", self.config.heave_frequency)
        phase = self._phases["heave"]
        heave = amp * math.sin(2.0 * math.pi * freq * t + phase)
        heave_rate = amp * 2.0 * math.pi * freq * math.cos(
            2.0 * math.pi * freq * t + phase
        )
        return heave, heave_rate

    def _burble_effect(self, s: float) -> float:
        """Return the burble disturbance at distance ``s`` from the stern.

        Modelled as a negative Gaussian bump in space, so the disturbance
        peaks near ``burble_mean`` and decays away from it.
        """
        params = self._current_params
        strength = params.get("burble_strength", self.config.burble_strength)
        mean = params.get("burble_mean", self.config.burble_mean)
        std = params.get("burble_std", self.config.burble_std)
        return -strength * math.exp(-((s - mean) ** 2) / (2.0 * max(std, 1e-6) ** 2))

    @staticmethod
    def _deck_height(s: float, heave: float, pitch: float) -> float:
        """Deck height relative to calm sea level at longitudinal position ``s``.

        The ship's deck is modelled as a rigid line that both pitches
        (rotates) and heaves (moves up and down). We approximate the deck
        height at distance ``s`` from the stern by:

            height(s) = heave + s * tan(c)
        """
        return heave + 5 * math.tan(pitch)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def outcome(self) -> Optional[EpisodeOutcome]:
        """Return the outcome object for the *last* completed episode."""
        return self._outcome

    @property
    def current_randomization(self) -> Dict[str, float]:
        """Return the randomization parameters used for the current episode."""
        return dict(self._current_params)
