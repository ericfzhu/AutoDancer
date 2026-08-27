"""Decide the predeclared EXP-0019 frozen player8 transfer gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autodancer.training.player10_transfer_compare import compare_assistance_transfer

MODES = (
    ("deterministic", "deterministic", 0),
    ("stochastic-106001", "stochastic", 106001),
    ("stochastic-106002", "stochastic", 106002),
)


def compare_player8_transfer(root: Path) -> dict[str, Any]:
    return compare_assistance_transfer(
        root,
        experiment_id="EXP-0019",
        profile="boss1hp-player8",
        modes=MODES,
        pass_decision="accept_player8_bridge_without_training",
        fail_decision="run_mixed_player8_replay",
        selected_checkpoint="runs/assisted-death-metal/training/seed-68002/final.pt",
        minimum_sampled_completion=0.6,
        minimum_distinct_successful_seeds=16,
        maximum_sampled_death=0.4,
        minimum_deterministic_completion=1 / 3,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide EXP-0019 player8 gate")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(compare_player8_transfer(arguments.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
