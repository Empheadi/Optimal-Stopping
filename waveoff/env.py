"""Environment model for the carrier wave-off optimal stopping problem."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np


ACTION_CONTINUE = 0
ACTION_WAVE_OFF = 1


@dataclass
class WaveOffProfile:
    """Parameters describing the fixed wave-off climb profile."""

    commanded_climb_rate: float = 8.0  # m/s
    engine_time_constant: float = 1.0  # s
    max_vertical_accel: float = 3.0  # m/s^2


@dataclass
class NoiseParams:
    """Noise parameters for glide-slope error dynamics."""

    eta_std: float = 0.2
    nu_std: float = 0.05


@dataclass
class EnvironmentConfig:
    """Configuration for the wave-off optimal stopping environment."""

    dt: float = 0.1
    gamma: float = 0.99
    glide_slope_deg: float = 3.5
    approach_speed: float = 65.0  # m/s (~126 knots)
    s_init: float = 1200.0  # initial distance to stern (m)
    s_offset: float = 50.0  # offset so that ref height > 0 initially
    trap_distance: float = 75.0  # distance past stern at which landing is evaluated (m)
    target_wire: float = 53.6  # ideal touchdown distance past stern (m)
    landing_tolerance: float = 12.0  # tolerance scale for landing reward (m)
    ramp_clearance: float = 3.0  # minimum clearance (m)
    stern_clearance: float = 6.0  # clearance needed during wave-off (m)
    max_time: float = 80.0  # safety cutoff to avoid infinite loops
    phi: float = 0.95  # AR coefficient for error
    rho: float = 0.85  # AR coefficient for error rate
    psi: float = 0.05  # coupling between rate and error
    beta_burble: float = 0.3  # coupling from burble disturbance
    beta_pitch: float = 0.5  # coupling from deck pitch rate
    initial_error_mean: float = 0.5
    initial_error_std: float = 0.75
    initial_rate_std: float = 0.4
    heave_amplitude: float = 0.2
    heave_frequency: float = 0.5
    pitch_amplitude: float = math.radians(1.5)
    pitch_frequency: float = 0.8
    burble_strength: float = 2.0
    burble_mean: float = 40.0
    burble_std: float = 25.0
    wave_off_profile: WaveOffProfile = WaveOffProfile()
    noise: NoiseParams = NoiseParams()


@dataclass
class RandomizationParams:
    """Randomization ranges for domain randomization."""

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
    """Information captured at the end of an episode."""

    mode: str
    touchdown_error: Optional[float]
    stern_clearance: Optional[float]
    randomization: Dict[str, float]


class WaveOffEnv:
    """Wave-off decision environment with simplified glide-path dynamics."""

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        randomization: Optional[RandomizationParams] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or EnvironmentConfig()
        self.randomization = randomization or RandomizationParams()
        self.rng = np.random.default_rng(seed)
        self._state: Optional[np.ndarray] = None
        self._time = 0.0
        self._waved_off = False
        self._wave_off_vz = 0.0
        self._outcome: Optional[EpisodeOutcome] = None
        self._current_params: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        randomization: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Reset the environment.

        Args:
            seed: Optional seed for numpy RNG.
            randomization: Optional dictionary overriding sampled parameters.

        Returns:
            Initial observation vector.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self.config
        params = self._sample_randomization(randomization)
        self._current_params = params

        e0 = self.rng.normal(cfg.initial_error_mean, cfg.initial_error_std)
        e_dot0 = self.rng.normal(0.0, cfg.initial_rate_std)
        s0 = cfg.s_init
        pitch_phase = self.rng.uniform(0, 2 * math.pi)
        heave_phase = self.rng.uniform(0, 2 * math.pi)
        self._phases = {"pitch": pitch_phase, "heave": heave_phase}

        self._time = 0.0
        self._waved_off = False
        approach_speed = params.get("approach_speed", cfg.approach_speed)
        self._wave_off_vz = -approach_speed * math.sin(math.radians(cfg.glide_slope_deg))
        pitch, pitch_rate = self._compute_pitch(0.0)
        heave, _ = self._compute_heave(0.0)
        burble = self._burble_effect(s0)

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
        """Advance the simulation by one step."""
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if action not in (ACTION_CONTINUE, ACTION_WAVE_OFF):
            raise ValueError(f"Invalid action: {action}")

        cfg = self.config
        params = self._current_params
        e, e_dot, s, v_z, _, _, _, _ = self._state
        dt = cfg.dt

        # Update along-deck position
        v_app = params.get("approach_speed", cfg.approach_speed)
        s_next = s - v_app * dt
        self._time += dt

        # Determine if wave-off was commanded this step
        if action == ACTION_WAVE_OFF and not self._waved_off:
            self._waved_off = True

        # Error rate dynamics
        eta = self.rng.normal(0.0, params.get("eta_std", cfg.noise.eta_std))
        nu = self.rng.normal(0.0, params.get("nu_std", cfg.noise.nu_std))
        e_dot_next = cfg.rho * e_dot + nu

        burble = self._burble_effect(s)
        pitch, pitch_rate = self._compute_pitch(self._time)
        heave, _ = self._compute_heave(self._time)

        e_next = (
            cfg.phi * e
            + cfg.psi * e_dot * dt
            + cfg.beta_burble * burble * dt
            + cfg.beta_pitch * pitch_rate * dt
            + eta
        )

        glide_rad = math.radians(cfg.glide_slope_deg)
        z_ref = math.tan(glide_rad) * (s_next + cfg.s_offset)
        deck_height = self._deck_height(s_next, heave, pitch)
        z_next = e_next + deck_height + (z_ref - self._deck_height(s_next, 0.0, 0.0))

        # Wave-off vertical dynamics
        wave_profile = cfg.wave_off_profile
        if self._waved_off:
            accel = min(
                wave_profile.max_vertical_accel,
                (wave_profile.commanded_climb_rate - v_z)
                / max(wave_profile.engine_time_constant, 1e-6),
            )
            v_z_next = v_z + accel * dt
        else:
            # Maintain vertical speed implied by glide slope
            v_z_next = -v_app * math.sin(glide_rad)

        next_burble = self._burble_effect(s_next)
        obs = np.array(
            [
                e_next,
                e_dot_next,
                s_next,
                v_z_next,
                pitch,
                pitch_rate,
                heave,
                next_burble,
            ],
            dtype=np.float32,
        )

        reward = 0.0
        cost = 0
        done = False
        info: Dict[str, float] = {}

        def deck_relative_height(s_val: float, z_val: float) -> float:
            deck = self._deck_height(s_val, heave, pitch)
            return z_val - deck

        # Ramp strike / stern checks
        if not self._waved_off and s_next <= 0.0:
            clearance = deck_relative_height(0.0, z_next)
            info["stern_clearance"] = clearance
            if clearance < cfg.ramp_clearance:
                reward = -100.0
                cost = 1
                done = True
                outcome = EpisodeOutcome(
                    mode="ramp_strike",
                    touchdown_error=None,
                    stern_clearance=clearance,
                    randomization=params,
                )
                self._outcome = outcome

        # Landing evaluation once past target wire
        if not done and not self._waved_off and s_next <= -cfg.trap_distance:
            touchdown_x = -s_next
            touchdown_error = touchdown_x - cfg.target_wire
            reward = 1.0 - min(1.0, abs(touchdown_error) / cfg.landing_tolerance)
            done = True
            outcome = EpisodeOutcome(
                mode="landed",
                touchdown_error=touchdown_error,
                stern_clearance=info.get("stern_clearance"),
                randomization=params,
            )
            self._outcome = outcome

        # Wave-off evaluation when crossing stern
        if not done and self._waved_off and s_next <= 0.0:
            clearance = deck_relative_height(0.0, z_next)
            info["stern_clearance"] = clearance
            if clearance >= cfg.stern_clearance:
                reward = 0.2 + 0.02 * max(0.0, clearance - cfg.stern_clearance)
                outcome = EpisodeOutcome(
                    mode="wave_off_success",
                    touchdown_error=None,
                    stern_clearance=clearance,
                    randomization=params,
                )
            else:
                reward = -100.0
                cost = 1
                outcome = EpisodeOutcome(
                    mode="wave_off_failure",
                    touchdown_error=None,
                    stern_clearance=clearance,
                    randomization=params,
                )
            done = True
            self._outcome = outcome

        # Emergency termination if exceeding max time or far past trap distance
        if not done and self._time >= cfg.max_time:
            done = True
            reward = -5.0
            outcome = EpisodeOutcome(
                mode="timeout",
                touchdown_error=None,
                stern_clearance=info.get("stern_clearance"),
                randomization=params,
            )
            self._outcome = outcome

        self._state = obs.copy()
        return obs, float(reward), cost, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_randomization(
        self, overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        params: Dict[str, float] = {}
        ranges = asdict(self.randomization)
        for name, (low, high) in ranges.items():
            params[name] = self.rng.uniform(low, high)
        if overrides:
            params.update(overrides)
        return params

    def _compute_pitch(self, t: float) -> Tuple[float, float]:
        params = self._current_params
        amp = params.get("pitch_amplitude", self.config.pitch_amplitude)
        freq = params.get("pitch_frequency", self.config.pitch_frequency)
        phase = self._phases["pitch"]
        pitch = amp * math.sin(2.0 * math.pi * freq * t + phase)
        pitch_rate = amp * 2.0 * math.pi * freq * math.cos(2.0 * math.pi * freq * t + phase)
        return pitch, pitch_rate

    def _compute_heave(self, t: float) -> Tuple[float, float]:
        params = self._current_params
        amp = params.get("heave_amplitude", self.config.heave_amplitude)
        freq = params.get("heave_frequency", self.config.heave_frequency)
        phase = self._phases["heave"]
        heave = amp * math.sin(2.0 * math.pi * freq * t + phase)
        heave_rate = amp * 2.0 * math.pi * freq * math.cos(2.0 * math.pi * freq * t + phase)
        return heave, heave_rate

    def _burble_effect(self, s: float) -> float:
        params = self._current_params
        strength = params.get("burble_strength", self.config.burble_strength)
        mean = params.get("burble_mean", self.config.burble_mean)
        std = params.get("burble_std", self.config.burble_std)
        return -strength * math.exp(-((s - mean) ** 2) / (2.0 * max(std, 1e-6) ** 2))

    @staticmethod
    def _deck_height(s: float, heave: float, pitch: float) -> float:
        """Deck height relative to calm sea level at longitudinal position s."""
        return heave + s * math.tan(pitch)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def outcome(self) -> Optional[EpisodeOutcome]:
        return self._outcome

    @property
    def current_randomization(self) -> Dict[str, float]:
        """Return the randomization parameters used for the current episode."""
        return dict(self._current_params)

