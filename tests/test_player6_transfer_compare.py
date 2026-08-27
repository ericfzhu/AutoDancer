from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.player6_transfer_compare import (
    MODES,
    compare_player6_transfer,
)


def test_player6_transfer_gate_uses_normal_health_profile(tmp_path: Path) -> None:
    seeds = list(range(72001, 72025))
    (tmp_path / "heldout-selection.json").write_text(
        json.dumps({"seeds": seeds}), encoding="utf-8"
    )
    for name, mode, policy_seed in MODES:
        directory = tmp_path / name
        directory.mkdir()
        successes = 5 if mode == "deterministic" else 14
        results = [
            {
                "seed": seed,
                "status": "curriculum_complete" if index < successes else "dead",
                "furthest_zone": 2 if index < successes else 1,
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
                    "curriculum_profile": "boss1hp-player6",
                    "trained": {"results": results},
                }
            ),
            encoding="utf-8",
        )
    result = compare_player6_transfer(tmp_path)
    assert result["passed"] is True
    assert result["experiment_id"] == "EXP-0018"
    assert result["decision"] == "retain_parent_and_advance_to_boss_health"
