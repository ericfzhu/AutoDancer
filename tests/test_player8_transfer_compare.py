from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.player8_transfer_compare import (
    MODES,
    compare_player8_transfer,
)


def _write_gate(root: Path, successes: tuple[int, int, int]) -> None:
    seeds = list(range(75001, 75025))
    root.mkdir(parents=True, exist_ok=True)
    (root / "heldout-selection.json").write_text(
        json.dumps({"seeds": seeds}), encoding="utf-8"
    )
    for (name, mode, policy_seed), success_count in zip(MODES, successes, strict=True):
        directory = root / name
        directory.mkdir(exist_ok=True)
        results = [
            {
                "seed": seed,
                "status": "curriculum_complete" if index < success_count else "dead",
                "furthest_zone": 2 if index < success_count else 1,
                "boss_kills": 1 if index < success_count else 0,
                "boss_damage": 1 if index < success_count else 0,
            }
            for index, seed in enumerate(seeds)
        ]
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "controller_valid": True,
                    "infrastructure_events": [],
                    "worker_restarts": 0,
                    "policy_mode": mode,
                    "policy_seed": policy_seed,
                    "curriculum_start_level": 4,
                    "curriculum_target_level": 5,
                    "curriculum_profile": "boss1hp-player8",
                    "trained": {"results": results},
                }
            ),
            encoding="utf-8",
        )


def test_player8_gate_uses_stricter_predeclared_thresholds(tmp_path: Path) -> None:
    root = tmp_path / "gate"
    _write_gate(root, (8, 15, 14))
    failed = compare_player8_transfer(root)
    assert failed["sampled_aggregate"]["completion_rate"] == 29 / 48
    assert failed["passed"] is False
    assert failed["decision"] == "run_mixed_player8_replay"

    _write_gate(root, (8, 15, 15))
    passed = compare_player8_transfer(root)
    assert passed["sampled_aggregate"]["completion_rate"] == 30 / 48
    assert len(passed["sampled_aggregate"]["distinct_successful_seeds"]) == 15
    assert passed["passed"] is False

    _write_gate(root, (8, 16, 16))
    passed = compare_player8_transfer(root)
    assert passed["passed"] is True
    assert passed["decision"] == "accept_player8_bridge_without_training"
    assert passed["thresholds"]["minimum_sampled_completion"] == 0.6
