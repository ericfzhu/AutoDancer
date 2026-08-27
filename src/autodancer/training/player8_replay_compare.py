"""Compare the conditional EXP-0019 mixed player8 replay trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autodancer.training.assistance_replay_compare import compare_assistance_replay

CANDIDATES = ("parent", "seed-76001", "seed-76002", "seed-76003")
PROFILES = ("boss1hp-player8", "boss1hp-player10")
MODES = (
    ("deterministic", "deterministic", 0),
    ("stochastic-107001", "stochastic", 107001),
    ("stochastic-107002", "stochastic", 107002),
)


def compare_player8_replay(root: Path) -> dict[str, Any]:
    return compare_assistance_replay(
        root,
        experiment_id="EXP-0019",
        candidates=CANDIDATES,
        profiles=PROFILES,
        modes=MODES,
        primary_profile="boss1hp-player8",
        replay_profile="boss1hp-player10",
        primary_label="player8",
        expected_training_steps=51200,
        minimum_mean_completion=0.6,
        minimum_individual_completion=0.6,
        maximum_individual_death=0.4,
        minimum_replay_retention=0.8,
        pass_decision="advance_to_player6",
        fail_decision="retain_parent",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EXP-0019 replay trials")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(compare_player8_replay(arguments.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
