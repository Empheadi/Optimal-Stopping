"""Off-policy evaluation utilities for the wave-off project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .data import OfflineDataset


def _discounted_return(rewards: List[float], gamma: float) -> float:
    total = 0.0
    discount = 1.0
    for r in rewards:
        total += discount * r
        discount *= gamma
    return total


@dataclass
class OPEReport:
    expected_return: float
    return_lower_confidence: float
    failure_rate: float
    failure_upper_confidence: float


def evaluate_dataset_statistics(
    dataset: OfflineDataset, gamma: float = 0.99, delta: float = 1e-3
) -> OPEReport:
    if dataset.episode_ids is None:
        rewards = _discounted_return(dataset.rewards.tolist(), gamma)
        failure_rate = float(np.mean(dataset.costs > 0.5))
        return OPEReport(rewards, rewards, failure_rate, failure_rate)

    episodes = {}
    for idx, ep in enumerate(dataset.episode_ids):
        bucket = episodes.setdefault(int(ep), {"rewards": [], "costs": []})
        bucket["rewards"].append(float(dataset.rewards[idx]))
        bucket["costs"].append(float(dataset.costs[idx]))

    returns = []
    failures = []
    for values in episodes.values():
        returns.append(_discounted_return(values["rewards"], gamma))
        failures.append(float(any(cost > 0.5 for cost in values["costs"])))

    mean_return = float(np.mean(returns))
    mean_failure = float(np.mean(failures))

    lcb_return = _hoeffding_bound(returns, delta / 2.0, lower=True)
    ucb_failure = _hoeffding_bound(failures, delta / 2.0, lower=False)
    return OPEReport(mean_return, lcb_return, mean_failure, ucb_failure)


def _hoeffding_bound(values: List[float], delta: float, lower: bool) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    mean = float(arr.mean())
    value_range = arr.max() - arr.min()
    if value_range <= 0.0:
        return mean
    bound = value_range * np.sqrt(np.log(1.0 / max(delta, 1e-12)) / (2.0 * arr.size))
    return mean - bound if lower else mean + bound

