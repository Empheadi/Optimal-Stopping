"""Conservative Q-Learning trainer for the wave-off problem."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import OfflineDataset
from tqdm import tqdm

# We assume the standard action encoding used everywhere else:
# 0 -> CONTINUE, 1 -> WAVE_OFF.
ACTION_CONTINUE = 0
ACTION_WAVE_OFF = 1


@dataclass
class CQLConfig:
    dataset: Path
    output_dir: Path
    hidden_size: int = 128
    hidden_layers: int = 3
    batch_size: int = 256

    # "Epochs" now just group some number of replay updates
    epochs: int = 200

    gamma: float = 0.99
    alpha: float = 1.0
    lr: float = 3e-4
    target_update_period: int = 200
    device: str = "cpu"

    # Safety Lagrangian
    lambda_lr: float = 0.01
    safety_threshold: float = 5e-3

    # Wave-off window (must be consistent with EnvironmentConfig)
    wave_off_s_min: float = 100.0
    wave_off_s_max: float = 600.0

    # Replay-style training: how many gradient steps per epoch.
    # If 0, we default to roughly one "full pass" worth of steps:
    # steps_per_epoch = num_samples // batch_size.
    steps_per_epoch: int = 0


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

        # Replay buffer size
        self.num_samples = self.dataset.observations.shape[0]

        # Network setup
        obs_dim = self.dataset.observations.shape[1]
        self.q_net = QNetwork(obs_dim, config.hidden_size, config.hidden_layers).to(self.device)
        self.target_net = QNetwork(obs_dim, config.hidden_size, config.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)

        # Lagrange multiplier for safety cost
        self.lambda_penalty = torch.tensor(0.0, device=self.device)

        # How many replay updates per epoch
        if config.steps_per_epoch > 0:
            self.steps_per_epoch = config.steps_per_epoch
        else:
            # Default: approximately one "full pass" worth of updates
            self.steps_per_epoch = max(1, self.num_samples // config.batch_size)

    # ------------------------------------------------------------------
    # Replay-buffer style batch sampling
    # ------------------------------------------------------------------
    def _sample_batch(self) -> dict:
        """Sample a mini-batch from the offline replay buffer."""
        batch_size = self.config.batch_size
        idx = np.random.randint(0, self.num_samples, size=batch_size)

        # Numpy -> torch, then to device
        obs = torch.from_numpy(self.dataset.observations[idx]).to(self.device)
        actions = torch.from_numpy(self.dataset.actions[idx]).long().to(self.device)
        rewards = torch.from_numpy(self.dataset.rewards[idx]).to(self.device)
        costs = torch.from_numpy(self.dataset.costs[idx]).to(self.device)
        next_obs = torch.from_numpy(self.dataset.next_observations[idx]).to(self.device)

        # dones are stored as bools; convert to float32
        dones_np = self.dataset.dones[idx].astype(np.float32)
        dones = torch.from_numpy(dones_np).to(self.device)

        return {
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "costs": costs,
            "next_observations": next_obs,
            "dones": dones,
        }

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Train CQL using the offline dataset as a replay buffer."""
        cfg = self.config

        for epoch in tqdm(range(cfg.epochs), desc="Training CQL"):
            epoch_loss = 0.0

            for step in range(self.steps_per_epoch):
                batch = self._sample_batch()
                loss = self._loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
                self.optimizer.step()

                epoch_loss += float(loss.item())

                # Periodic soft update of target network
                if (step + 1) % cfg.target_update_period == 0:
                    self._soft_update()

            # One more soft update at the end of the epoch
            self._soft_update()

            # Safety Lagrange multiplier update
            failure_prob = self.estimate_failure_probability()
            self.lambda_penalty = torch.clamp(
                self.lambda_penalty + cfg.lambda_lr * (failure_prob - cfg.safety_threshold),
                min=0.0,
            )

            avg_loss = epoch_loss / max(1, self.steps_per_epoch)
            print(
                f"Epoch {epoch+1}/{cfg.epochs}: "
                f"loss={avg_loss:.4f}, "
                f"lambda={self.lambda_penalty.item():.4f}, "
                f"fail_prob={failure_prob:.5f}"
            )

        self.save()

    def save(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.output_dir / "cql_policy.pt"
        torch.save({"model": self.q_net.state_dict(), "config": self.config.__dict__}, model_path)
        print(f"Saved model to {model_path}")

    # ------------------------------------------------------------------
    def _loss(self, batch: dict) -> torch.Tensor:
        """Standard CQL Bellman + conservative regularizer + safety cost."""
        cfg = self.config
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        costs = batch["costs"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]

        # Q(s,a)
        current_q = self.q_net(obs)
        q_taken = current_q.gather(1, actions.view(-1, 1)).squeeze(1)

        # Target: r - λ c + γ V(s')
        with torch.no_grad():
            next_q = self.target_net(next_obs)
            next_v = torch.logsumexp(next_q, dim=1)
            targets = rewards - self.lambda_penalty * costs + (1.0 - dones) * cfg.gamma * next_v

        bellman_error = F.mse_loss(q_taken, targets)

        # CQL conservative term: push up log-sum-exp Q and pull down Q(s,a)
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
        """Greedy policy with wave-off window enforcement.

        observations: array of shape (N, obs_dim) or (obs_dim,)
        where obs[2] is distance to stern s (as in env.Observation).
        """
        with torch.no_grad():
            obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)

            q_values = self.q_net(obs)
            actions = torch.argmax(q_values, dim=-1)  # 0 or 1

            # Enforce the wave-off window: outside [s_min, s_max], we force CONTINUE.
            # We rely on the standard observation layout: index 2 is distance to stern.
            s_values = obs[:, 2]
            mask_outside = (s_values < self.config.wave_off_s_min) | (
                s_values > self.config.wave_off_s_max
            )

            actions = actions.clone()
            actions[mask_outside] = ACTION_CONTINUE

            return actions.cpu().numpy()

    def estimate_failure_probability(self) -> float:
        """Estimate episode-level failure probability from the dataset.

        A failure is any episode that has cost > 0 at any time step.
        """
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


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/waveoff_dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("ckpts"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--safety-threshold", type=float, default=5e-3)

    # Wave-off window (must match env config)
    parser.add_argument(
        "--wave-off-s-min",
        type=float,
        default=100.0,
        help="Minimum distance to stern at which wave-off is allowed.",
    )
    parser.add_argument(
        "--wave-off-s-max",
        type=float,
        default=600.0,
        help="Maximum distance to stern at which wave-off is allowed.",
    )

    # Replay-style training control
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=2000,
        help=(
            "Number of replay gradient steps per epoch. "
            "If 0, defaults to roughly one full-pass worth of steps "
            "(num_samples // batch_size)."
        ),
    )

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
        wave_off_s_min=args.wave_off_s_min,
        wave_off_s_max=args.wave_off_s_max,
        steps_per_epoch=args.steps_per_epoch,
    )
    trainer = CQLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
