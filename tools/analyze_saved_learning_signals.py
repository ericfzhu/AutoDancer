"""Summarize saved first/last rollout signals without changing or selecting a policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from autodancer.constants import Action
from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.training.ppo import RolloutBatch, generalized_advantage_estimate


def summarize(path, gamma, gae_lambda):
    rollout = RolloutBatch(**torch.load(path, map_location="cpu", weights_only=False))
    advantages, targets = generalized_advantage_estimate(
        rollout.rewards,
        rollout.values,
        rollout.dones,
        rollout.next_value,
        terminations=rollout.terminations,
        truncation_values=rollout.truncation_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    by_action = {}
    for number in rollout.actions.unique().tolist():
        mask = rollout.actions == number
        by_action[Action(number).name] = {
            "count": int(mask.sum()),
            "mean_raw_advantage": float(advantages[mask].mean()),
            "mean_normalized_advantage": float(normalized[mask].mean()),
            "positive_normalized_fraction": float((normalized[mask] > 0).float().mean()),
        }
    return {
        "sha256": sha256_file(path),
        "transitions": rollout.actions.numel(),
        "nonzero_rewards": int((rollout.rewards != 0).sum()),
        "positive_rewards": int((rollout.rewards > 0).sum()),
        "negative_rewards": int((rollout.rewards < 0).sum()),
        "value_mean": float(rollout.values.mean()),
        "target_mean": float(targets.mean()),
        "raw_advantage_mean": float(advantages.mean()),
        "raw_advantage_std": float(advantages.std(unbiased=False)),
        "terminations": int(rollout.terminations.sum()),
        "truncations": int((rollout.dones.bool() & ~rollout.terminations.bool()).sum()),
        "by_action": by_action,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    root = parser.parse_args().run_dir.resolve()
    config = json.loads((root / "training/config.json").read_text())
    report = {
        "config_sha256": sha256_file(root / "training/config.json"),
        "batches": {
            str(update): summarize(
                root / f"rollout-update-{update:02d}.pt",
                config["ppo"]["gamma"],
                config["ppo"]["gae_lambda"],
            )
            for update in (1, 16)
        },
        "limitation": (
            "First and final batches contain different states and policies. Action groups "
            "are state-confounded; these descriptive statistics do not establish causal "
            "action utility or critic calibration. Bootstrap observations were not saved."
        ),
    }
    atomic_json(root / "saved-signal-analysis.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
