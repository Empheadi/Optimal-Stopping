# Optimal-Stopping

This repository hosts reference implementations for optimal stopping problems.
It now includes a simplified simulator and offline reinforcement learning tools
for the **carrier-based wave-off decision task** described in the problem
statement.

## Package layout

The new `waveoff` package bundles everything required to model, train and
evaluate the safe wave-off agent:

| Module | Description |
| --- | --- |
| `waveoff.env` | 2-D optimal-stopping environment with deck motion, burble, and wave-off dynamics. |
| `waveoff.baseline` | Handcrafted threshold policy used for dataset generation and comparisons. |
| `waveoff.gen_data` | Script to generate offline datasets with domain randomisation. |
| `waveoff.cql` | Conservative Q-Learning (CQL) trainer with a safety penalty. |
| `waveoff.ope` | Lightweight off-policy evaluation helpers producing confidence bounds. |
| `waveoff.acceptance` | Command-line tool to summarise OPE statistics for acceptance testing. |

## Quickstart

1. Create and activate a Python environment (3.9+ recommended).
2. Install dependencies: `pip install torch numpy` (and any extras you require).
3. Generate an offline dataset using the baseline policy:

   ```bash
   python -m waveoff.gen_data --episodes 4096 --output data/waveoff_dataset.npz
   ```

4. Train a conservative Q-learning policy:

   ```bash
   python -m waveoff.cql data/waveoff_dataset.npz runs/cql --epochs 100 --batch-size 512
   ```

5. Inspect offline evaluation statistics:

   ```bash
    python -m waveoff.acceptance data/waveoff_dataset.npz --delta 1e-4
   ```

These utilities are designed for experimentation and can be extended with more
advanced offline RL or safety analysis techniques as needed.

## License

This project is licensed under the MIT License — see the LICENSE file for
details.
