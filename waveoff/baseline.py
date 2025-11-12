"""Handcrafted baseline policy for the wave-off decision problem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .env import ACTION_CONTINUE, ACTION_WAVE_OFF


@dataclass
class BaselineThresholdPolicy:
    """Baseline rule-based policy for wave-off decisions."""

    error_threshold: float = -1.0
    closing_bias: float = 0.2
    rate_threshold: float = -0.6
    activation_distance: float = 120.0

    def __call__(self, observation: Iterable[float]) -> int:
        obs = np.asarray(observation, dtype=np.float32)
        e = float(obs[0])
        e_rate = float(obs[1])
        s = float(obs[2])

        dynamic_threshold = self.error_threshold + self.closing_bias * np.sqrt(
            max(0.0, self.activation_distance - s)
        )
        if e < dynamic_threshold:
            return ACTION_WAVE_OFF
        if e_rate < self.rate_threshold:
            return ACTION_WAVE_OFF
        return ACTION_CONTINUE

