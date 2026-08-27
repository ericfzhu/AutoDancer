from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.player10_transfer_compare import (
    MODES,
    compare_player10_transfer,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_player10_transfer_gate_uses_sampled_gameplay_and_deterministic_floor(tmp_path) -> None:
    seeds = list(range(69001, 69025))
    write_json(tmp_path / "heldout-selection.json", {"seeds": seeds})
    success_counts = {"deterministic": 5, "stochastic-100001": 13, "stochastic-100002": 12}
    for name, mode, policy_seed in MODES:
        successes = success_counts[name]
        results = [
            {
                "seed": seed,
                "status": "curriculum_complete" if index < successes else "dead",
                "furthest_zone": 2 if index < successes else 1,
                "furthest_floor": 1 if index < successes else 4,
                "boss_damage": int(index < successes),
                "boss_kills": int(index < successes),
            }
            for index, seed in enumerate(seeds)
        ]
        write_json(
            tmp_path / name / "report.json",
            {
                "controller_valid": True,
                "infrastructure_events": [],
                "worker_restarts": 0,
                "policy_mode": mode,
                "policy_seed": policy_seed,
                "curriculum_start_level": 4,
                "curriculum_target_level": 5,
                "curriculum_profile": "boss1hp-player10",
                "trained": {"results": results},
            },
        )
    result = compare_player10_transfer(tmp_path)
    assert result["passed"] is True
    assert result["decision"] == "retain_parent_and_advance_to_player6"
    assert result["sampled_aggregate"]["successes"] == 25
    assert len(result["sampled_aggregate"]["distinct_successful_seeds"]) == 13


def test_player10_transfer_gate_fails_when_argmax_has_no_success(tmp_path) -> None:
    seeds = list(range(69001, 69025))
    write_json(tmp_path / "heldout-selection.json", {"seeds": seeds})
    for name, mode, policy_seed in MODES:
        successes = 0 if mode == "deterministic" else 24
        results = [
            {
                "seed": seed,
                "status": "curriculum_complete" if index < successes else "dead",
                "furthest_zone": 2 if index < successes else 1,
            }
            for index, seed in enumerate(seeds)
        ]
        write_json(
            tmp_path / name / "report.json",
            {
                "controller_valid": True,
                "infrastructure_events": [],
                "worker_restarts": 0,
                "policy_mode": mode,
                "policy_seed": policy_seed,
                "curriculum_start_level": 4,
                "curriculum_target_level": 5,
                "curriculum_profile": "boss1hp-player10",
                "trained": {"results": results},
            },
        )
    assert compare_player10_transfer(tmp_path)["passed"] is False
