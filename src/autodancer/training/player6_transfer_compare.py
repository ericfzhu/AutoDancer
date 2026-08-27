"""Decide the predeclared EXP-0018 normal-health transfer gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autodancer.training.player10_transfer_compare import compare_assistance_transfer

MODES = (
    ("deterministic", "deterministic", 0),
    ("stochastic-103001", "stochastic", 103001),
    ("stochastic-103002", "stochastic", 103002),
)


def compare_player6_transfer(root: Path) -> dict[str, Any]:
    return compare_assistance_transfer(
        root,
        experiment_id="EXP-0018",
        profile="boss1hp-player6",
        modes=MODES,
        pass_decision="retain_parent_and_advance_to_boss_health",
        fail_decision="run_mixed_player6_replay",
        selected_checkpoint="runs/assisted-death-metal/training/seed-68002/final.pt",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide EXP-0018 normal-health gate")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(compare_player6_transfer(arguments.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
