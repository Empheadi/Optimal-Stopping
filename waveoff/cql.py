"""Conservative Q-Learning trainer for the wave-off problem."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import OfflineDataset


@dataclass
class CQLConfig:
    dataset: Path
    output_dir: Path
    hidden_size: int = 128
    hidden_layers: int = 3
    batch_size: int = 256
    epochs: int = 200
    gamma: float = 0.99
    alpha: float = 1.0
    lr: float = 3e-4
    target_update_period: int = 200
    device: str = "cpu"
    lambda_lr: float = 0.01
    safety_threshold: float = 5e-3


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, hidden_layers: int) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.ReLU())
            last_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.backbone(x)
        return self.head(z)


class CQLTrainer:
    def __init__(self, config: CQLConfig) -> None:
        self.config = config
        self.dataset = OfflineDataset(config.dataset)
        self.device = torch.device(config.device)
        obs_dim = self.dataset.observations.shape[1]
        self.q_net = QNetwork(obs_dim, config.hidden_size, config.hidden_layers).to(self.device)
        self.target_net = QNetwork(obs_dim, config.hidden_size, config.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.lambda_penalty = torch.tensor(0.0, device=self.device)

    # ------------------------------------------------------------------
    def train(self) -> None:
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(loader):
                loss = self._loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += float(loss.item())

                if (step + 1) % self.config.target_update_period == 0:
                    self._soft_update()

            self._soft_update()
            failure_prob = self.estimate_failure_probability()
            self.lambda_penalty = torch.clamp(
                self.lambda_penalty
                + self.config.lambda_lr * (failure_prob - self.config.safety_threshold),
                min=0.0,
            )
            avg_loss = epoch_loss / max(1, len(loader))
            print(
                f"Epoch {epoch+1}/{self.config.epochs}: loss={avg_loss:.4f}, "
                f"lambda={self.lambda_penalty.item():.4f}, fail_prob={failure_prob:.5f}"
            )

        self.save()

    def save(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.output_dir / "cql_policy.pt"
        torch.save({"model": self.q_net.state_dict(), "config": self.config.__dict__}, model_path)
        print(f"Saved model to {model_path}")

    # ------------------------------------------------------------------
    def _loss(self, batch: dict) -> torch.Tensor:
        cfg = self.config
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        costs = batch["costs"].to(self.device)
        next_obs = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)

        current_q = self.q_net(obs)
        q_taken = current_q.gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs)
            next_v = torch.logsumexp(next_q, dim=1)
            targets = rewards - self.lambda_penalty * costs + (1.0 - dones) * cfg.gamma * next_v

        bellman_error = F.mse_loss(q_taken, targets)
        cql_reg = cfg.alpha * (
            torch.logsumexp(current_q, dim=1).mean() - q_taken.mean()
        )
        return bellman_error + cql_reg

    def _soft_update(self, tau: float = 0.05) -> None:
        with torch.no_grad():
            for param, target in zip(self.q_net.parameters(), self.target_net.parameters()):
                target.data.mul_(1.0 - tau).add_(tau * param.data)

    # ------------------------------------------------------------------
    def policy(self, observations: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            q_values = self.q_net(obs)
            return torch.argmax(q_values, dim=-1).cpu().numpy()

    def estimate_failure_probability(self) -> float:
        if self.dataset.episode_ids is None:
            return float(self.dataset.costs.mean())
        episode_costs = {}
        for idx, episode in enumerate(self.dataset.episode_ids):
            episode_costs.setdefault(int(episode), 0.0)
            episode_costs[int(episode)] = max(
                episode_costs[int(episode)], float(self.dataset.costs[idx])
            )
        if not episode_costs:
            return float(self.dataset.costs.mean())
        return float(np.mean(list(episode_costs.values())))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to offline dataset (.npz)")
    parser.add_argument("output", type=Path, help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--safety-threshold", type=float, default=5e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CQLConfig(
        dataset=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        alpha=args.alpha,
        safety_threshold=args.safety_threshold,
    )
    trainer = CQLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
