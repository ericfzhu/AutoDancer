from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.boss_curriculum_compare import (
    ARMS,
    TRIALS,
    compare_boss_curriculum,
)


def _write_run(root: Path, arm: str, successful_modes: int, successful_seeds: int) -> None:
    training = root / "training" / arm
    training.mkdir(parents=True)
    records = [
        {
            "global_step": 30720,
            "updates": 30,
            "episodes": 8,
            "curriculum_completions": 0,
            "time_limits": 8,
            "episode_seeds": list(range(64001, 64009)),
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 1.0,
            "worker_restarts": 0,
        },
        {
            "global_step": 122880,
            "updates": 120,
            "episodes": 8,
            "curriculum_completions": 4,
            "time_limits": 2,
            "episode_seeds": list(range(64009, 64017)),
            "policy_loss": 0.05,
            "value_loss": 0.1,
            "entropy": 0.8,
            "worker_restarts": 0,
        },
    ]
    (training / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    for mode_index, (trial, policy_mode, policy_seed) in enumerate(TRIALS):
        directory = root / "evaluation" / arm / trial
        directory.mkdir(parents=True)
        successes = successful_seeds if mode_index < successful_modes else 0
        results = [
            {
                "seed": 66001 + index,
                "status": "curriculum_complete" if index < successes else "dead",
                "furthest_zone": 2 if index < successes else 1,
                "furthest_floor": 1 if index < successes else 4,
                "turns": 100 + index,
            }
            for index in range(6)
        ]
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "controller_valid": True,
                    "infrastructure_events": [],
                    "worker_restarts": 0,
                    "policy_mode": policy_mode,
                    "policy_seed": policy_seed,
                    "curriculum_start_level": 4,
                    "curriculum_target_level": 5,
                    "trained": {"results": results},
                }
            ),
            encoding="utf-8",
        )


def test_boss_curriculum_comparison_selects_only_repeatable_success(tmp_path: Path) -> None:
    _write_run(tmp_path, ARMS[0], successful_modes=1, successful_seeds=4)
    _write_run(tmp_path, ARMS[1], successful_modes=2, successful_seeds=3)

    result = compare_boss_curriculum(tmp_path)

    assert not result["arms"][ARMS[0]]["passed"]
    assert result["arms"][ARMS[1]]["passed"]
    assert result["selected_arm"] == ARMS[1]
    assert result["decision"] == "advance_to_floor3_curriculum"
    assert result["normal_start_promotable"] is False


def test_boss_curriculum_comparison_requires_real_heldout_success(tmp_path: Path) -> None:
    for arm in ARMS:
        _write_run(tmp_path, arm, successful_modes=0, successful_seeds=0)

    result = compare_boss_curriculum(tmp_path)

    assert result["selected_arm"] is None
    assert result["extension_arm"] is None
    assert result["decision"] == "declare_boss_tactical_action_intervention"
