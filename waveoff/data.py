"""Utility structures for offline datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class OfflineDataset(Dataset[Dict[str, torch.Tensor]]):
    """Offline dataset backed by an `.npz` file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        data = np.load(self.path)
        self.observations = data["observations"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.rewards = data["rewards"].astype(np.float32)
        self.costs = data["costs"].astype(np.float32)
        self.next_observations = data["next_observations"].astype(np.float32)
        self.dones = data["dones"].astype(np.bool_)
        self.importance_weights = data.get("is_weights", np.ones_like(self.rewards))
        self.episode_ids = data.get("episode_ids")

    def __len__(self) -> int:  # type: ignore[override]
        return self.observations.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "observations": torch.from_numpy(self.observations[idx]),
            "actions": torch.tensor(self.actions[idx], dtype=torch.long),
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32),
            "costs": torch.tensor(self.costs[idx], dtype=torch.float32),
            "next_observations": torch.from_numpy(self.next_observations[idx]),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32),
            "importance_weights": torch.tensor(
                self.importance_weights[idx], dtype=torch.float32
            ),
        }

    def iter_episodes(self) -> Iterator[Tuple[Dict[str, object], Dict[str, float]]]:
        meta_path = self.path.with_suffix(".meta.json")
        if not meta_path.exists():
            return iter(())
        meta = json.loads(meta_path.read_text())
        outcomes = meta.get("outcomes", [])
        randomizations = meta.get("randomizations", [])
        return iter(zip(outcomes, randomizations))


