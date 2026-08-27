from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.curriculum_compare import TRIALS, compare_curriculum_feasibility


def _write_report(path: Path, successes: set[int]) -> None:
    results = [
        {
            "seed": seed,
            "status": "curriculum_complete" if seed in successes else "dead",
            "furthest_zone": 2 if seed in successes else 1,
            "furthest_floor": 1 if seed in successes else 4,
            "turns": 20,
            "enemy_kills": 1,
            "player_damage": 1,
        }
        for seed in range(63001, 63005)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "controller_valid": True,
                "curriculum_start_level": 4,
                "curriculum_target_level": 5,
                "trained": {"results": results},
            }
        ),
        encoding="utf-8",
    )


def test_curriculum_comparison_requires_three_seeds_and_two_modes(tmp_path: Path) -> None:
    _write_report(tmp_path / "evaluation" / TRIALS[0] / "report.json", {63001, 63002})
    _write_report(tmp_path / "evaluation" / TRIALS[1] / "report.json", {63002, 63003})
    _write_report(tmp_path / "evaluation" / TRIALS[2] / "report.json", set())
    result = compare_curriculum_feasibility(tmp_path)
    assert result["distinct_successful_seeds"] == [63001, 63002, 63003]
    assert result["successful_policy_modes"] == 2
    assert result["existing_boss_competence_supported"] is True
    assert result["normal_start_promotable"] is False
