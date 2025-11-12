"""Acceptance test harness using offline evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import OfflineDataset
from .ope import evaluate_dataset_statistics


def run_acceptance(dataset_path: Path, delta: float) -> None:
    dataset = OfflineDataset(dataset_path)
    report = evaluate_dataset_statistics(dataset, gamma=0.99, delta=delta)
    print("=== Offline Evaluation Report ===")
    print(f"Expected return (mean): {report.expected_return:.3f}")
    print(
        f"Return lower confidence bound (1-{delta:.1e}): "
        f"{report.return_lower_confidence:.3f}"
    )
    print(f"Failure rate (mean): {report.failure_rate:.5f}")
    print(
        f"Failure upper confidence bound (1-{delta:.1e}): "
        f"{report.failure_upper_confidence:.5f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Offline dataset (.npz)")
    parser.add_argument("--delta", type=float, default=1e-3, help="Confidence level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_acceptance(args.dataset, args.delta)


if __name__ == "__main__":
    main()
