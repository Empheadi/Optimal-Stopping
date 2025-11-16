"""Practical optimal-stopping trainer for the carrier wave-off task."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .env import ACTION_CONTINUE, ACTION_WAVE_OFF, WaveOffEnv


@dataclass
class OptimalStopConfig:
    episodes: int = 4096
    seed: int = 0
    gamma: float = 0.99
    batch_size: int = 512
    epochs: int = 30
    hidden_size: int = 64
    hidden_layers: int = 2
    lr: float = 3e-4
    eval_episodes: int = 512
    device: str = "cpu"
    output: Path = Path("runs/optimal_stop")


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, hidden_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last, hidden_size))
            layers.append(nn.ReLU())
            last = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if len(self.backbone) > 0:
            z = self.backbone(x)
        else:
            z = x
        return self.head(z)


@dataclass
class Trajectory:
    observations: List[np.ndarray]
    rewards: List[float]
    stop_values: List[float]


class BranchSimulator:
    """Helper that evaluates counterfactual actions via state snapshots."""

    def __init__(self, base_env: WaveOffEnv) -> None:
        self.base_env = base_env
        self.branch_env = WaveOffEnv(config=base_env.config, randomization=base_env.randomization)

    def evaluate(self, snapshot: Dict[str, object], action: int, gamma: float) -> float:
        self.branch_env.set_internal_state(snapshot)
        total = 0.0
        discount = 1.0
        done = False
        while not done:
            _, reward, _cost, done, _ = self.branch_env.step(action)
            total += discount * reward
            discount *= gamma
        return total


def collect_trajectories(env: WaveOffEnv, config: OptimalStopConfig) -> List[Trajectory]:
    rng = np.random.default_rng(config.seed)
    env.seed(int(rng.integers(0, 2**32 - 1)))
    oracle = BranchSimulator(env)
    trajectories: List[Trajectory] = []
    for _ in range(config.episodes):
        obs = env.reset()
        traj = Trajectory(observations=[], rewards=[], stop_values=[])
        done = False
        while not done:
            snapshot = env.get_internal_state()
            s = float(obs[2])
            if env.config.wave_off_s_min <= s <= env.config.wave_off_s_max:
                stop_val = oracle.evaluate(snapshot, ACTION_WAVE_OFF, config.gamma)
            else:
                stop_val = float("-inf")
            traj.observations.append(obs.copy())
            traj.stop_values.append(stop_val)

            obs, reward, _cost, done, _ = env.step(ACTION_CONTINUE)
            traj.rewards.append(float(reward))
        trajectories.append(traj)
    return trajectories


def label_trajectory(traj: Trajectory, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(traj.rewards)
    actions = np.zeros(T, dtype=np.int64)
    values = np.zeros(T, dtype=np.float32)
    next_value = 0.0
    for t in reversed(range(T)):
        cont = traj.rewards[t] + gamma * next_value
        stop = traj.stop_values[t]
        if stop > cont:
            actions[t] = ACTION_WAVE_OFF
            values[t] = stop
            next_value = stop
        else:
            actions[t] = ACTION_CONTINUE
            values[t] = cont
            next_value = cont
    return actions, values


def build_dataset(trajectories: Sequence[Trajectory], gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    obs_list: List[np.ndarray] = []
    label_list: List[int] = []
    for traj in trajectories:
        actions, _values = label_trajectory(traj, gamma)
        obs_list.extend(traj.observations)
        label_list.extend(actions.tolist())
    observations = np.stack(obs_list, axis=0).astype(np.float32)
    labels = np.asarray(label_list, dtype=np.int64)
    return observations, labels


class OptimalStoppingTrainer:
    def __init__(self, config: OptimalStopConfig) -> None:
        self.config = config
        self.env = WaveOffEnv(seed=config.seed)
        self.device = torch.device(config.device)
        self.policy = PolicyNet(8, config.hidden_size, config.hidden_layers).to(self.device)

    def train(self) -> None:
        print("Collecting trajectories for dynamic programming labels...")
        trajectories = collect_trajectories(self.env, self.config)
        observations, labels = build_dataset(trajectories, self.config.gamma)
        dataset = TensorDataset(
            torch.from_numpy(observations), torch.from_numpy(labels)
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)
        loss_fn = nn.CrossEntropyLoss()

        self.policy.train()
        for epoch in range(self.config.epochs):
            running = 0.0
            for batch_obs, batch_labels in loader:
                batch_obs = batch_obs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                logits = self.policy(batch_obs)
                loss = loss_fn(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += float(loss.item())
            avg_loss = running / max(1, len(loader))
            print(f"Epoch {epoch+1}/{self.config.epochs}: loss={avg_loss:.4f}")

        self.save_model()
        self.evaluate()

    def save_model(self) -> None:
        self.config.output.mkdir(parents=True, exist_ok=True)
        model_path = self.config.output / "optimal_stop.pt"
        torch.save({"model": self.policy.state_dict(), "config": self.config.__dict__}, model_path)
        print(f"Saved model to {model_path}")

    def evaluate(self) -> None:
        print("Evaluating trained policy...")
        env = WaveOffEnv(seed=self.config.seed + 12345)
        stats = run_policy(env, self.policy, self.config.eval_episodes, self.device)
        total = sum(stats.values())
        for key, value in stats.items():
            rate = value / max(1, total)
            print(f"{key:20s}: {value:6d} ({rate:.3f})")


def run_policy(env: WaveOffEnv, policy: PolicyNet, episodes: int, device: torch.device) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = select_action(policy, obs, env, device)
            obs, _reward, _cost, done, _ = env.step(action)
        outcome = env.outcome
        if outcome is None:
            key = "unknown"
        else:
            key = outcome.mode
        stats[key] = stats.get(key, 0) + 1
    return stats


def select_action(policy: PolicyNet, obs: np.ndarray, env: WaveOffEnv, device: torch.device) -> int:
    s = float(obs[2])
    if s < env.config.wave_off_s_min or s > env.config.wave_off_s_max:
        return ACTION_CONTINUE
    policy.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0)
        logits = policy(tensor)
        action = int(torch.argmax(logits, dim=1).item())
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/optimal_stop"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = OptimalStopConfig(
        episodes=args.episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        eval_episodes=args.eval_episodes,
        device=args.device,
        output=args.output,
        seed=args.seed,
    )
    trainer = OptimalStoppingTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
