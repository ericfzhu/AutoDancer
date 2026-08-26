from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.action_contract_compare import (
    TRIALS,
    compare_action_contract,
)


def _write_report(
    root: Path,
    arm: str,
    trial: str,
    *,
    walls: int,
    limits: int,
    progress: float,
    kills: int,
    items: int,
) -> None:
    mode = "deterministic" if trial == "deterministic" else "stochastic"
    policy_seed = 0 if mode == "deterministic" else int(trial.rsplit("-", 1)[1])
    results = [
        {
            "seed": seed,
            "turns": 100,
            "status": "step_limit" if index < limits else "dead",
            "furthest_zone": 1,
            "furthest_floor": 2 if index == 0 else 1,
        }
        for index, seed in enumerate(range(61_001, 61_025))
    ]
    report = {
        "controller_valid": True,
        "worker_restarts": 0,
        "infrastructure_events": [],
        "policy_mode": mode,
        "policy_seed": policy_seed,
        "action_contract": "current" if arm == "current-11" else arm,
        "seeds": list(range(61_001, 61_025)),
        "trained": {
            "results": results,
            "action_outcome_counts": {"wall_attempt": walls},
            "mean_progress": progress,
            "furthest_zone": 1,
            "furthest_floor": 2,
            "enemy_kills": kills,
            "item_pickups": items,
            "known_invalid_wall_discoveries": walls,
            "mean_masked_directions": 0.2,
        },
    }
    path = root / "evaluation" / f"{arm}-{trial}" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_action_contract_comparison_passes_predeclared_thresholds(tmp_path: Path) -> None:
    for trial in TRIALS:
        _write_report(
            tmp_path,
            "current-11",
            trial,
            walls=1_000,
            limits=8,
            progress=1.1,
            kills=100,
            items=20,
        )
        _write_report(
            tmp_path,
            "known-invalid-wall-v1",
            trial,
            walls=200,
            limits=4,
            progress=1.2,
            kills=85,
            items=18,
        )

    comparison = compare_action_contract(tmp_path)

    assert comparison["passed"] is True
    assert comparison["decision"] == "authorize_controlled_training_comparison"
    assert comparison["wall_attempt_rate_reduction"] == 0.8


def test_action_contract_comparison_rejects_incomplete_evidence(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        "current-11",
        "deterministic",
        walls=1_000,
        limits=8,
        progress=1.1,
        kills=100,
        items=20,
    )

    comparison = compare_action_contract(tmp_path)

    assert comparison["passed"] is False
    assert comparison["decision"] == "invalid_or_incomplete_evidence"
