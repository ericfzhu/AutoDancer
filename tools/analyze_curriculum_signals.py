"""Describe saved direct-start and curriculum rewards and observed terminal returns."""

import json
from pathlib import Path

import torch

from autodancer.experiments.schema import atomic_json
from autodancer.training.ppo import RolloutBatch
from tools.analyze_saved_learning_signals import summarize
from tools.audit_saved_batch_stability import observed_terminal_returns


def main():
    root = Path(__file__).resolve().parents[1]
    result = {}
    for name, directory in (
        ("direct", "stable-navigation-learning-exp0041"),
        ("curriculum", "bounded-curriculum-transfer-exp0043"),
    ):
        run = root / "runs" / directory
        config = json.loads((run / "training/config.json").read_text())
        gamma, lam = config["ppo"]["gamma"], config["ppo"]["gae_lambda"]
        result[name] = {}
        for update in (1, 16):
            path = run / f"rollout-update-{update:02d}.pt"
            summary = summarize(path, gamma, lam)
            batch = RolloutBatch(**torch.load(path, map_location="cpu", weights_only=False))
            targets, mask = observed_terminal_returns(batch, gamma)
            summary["observed_terminal_subset"] = {
                "transitions": int(mask.sum()),
                "value_mean": float(batch.values[mask].mean()) if mask.any() else None,
                "return_mean": float(targets[mask].mean()) if mask.any() else None,
                "squared_error_mean": float(((targets[mask] - batch.values[mask]) ** 2).mean())
                if mask.any()
                else None,
            }
            result[name][str(update)] = summary
    result["limitations"] = (
        "Different state distributions and policies; no causal action-utility or global "
        "critic-calibration claim. Observed terminal subsets exclude unfinished suffixes and "
        "truncations and contain correlated transitions from selected completed trajectories. "
        "Returns include the configured potential shaping and task reward."
    )
    atomic_json(root / "runs/bounded-curriculum-transfer-exp0043/signal-comparison.json", result)
    print(
        json.dumps(
            {
                k: {u: {a: b for a, b in s.items() if a != "by_action"} for u, s in v.items()}
                if isinstance(v, dict)
                else v
                for k, v in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
