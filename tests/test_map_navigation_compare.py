from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.map_navigation_compare import TRIALS, compare_map_navigation


def _write(
    root: Path,
    arm: str,
    trial: str,
    *,
    progress: float,
    limits: int,
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
            "furthest_floor": 2 if index < 3 else 1,
        }
        for index, seed in enumerate(range(62_001, 62_025))
    ]
    report = {
        "controller_valid": True,
        "worker_restarts": 0,
        "infrastructure_events": [],
        "policy_mode": mode,
        "policy_seed": policy_seed,
        "action_contract": "current" if arm == "current-11" else arm,
        "seeds": list(range(62_001, 62_025)),
        "trained": {
            "results": results,
            "mean_progress": progress,
            "furthest_zone": 1,
            "furthest_floor": 2,
            "enemy_kills": kills,
            "item_pickups": items,
        },
    }
    path = root / "evaluation" / f"{arm}-{trial}" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_map_navigation_comparison_passes_declared_gate(tmp_path: Path) -> None:
    for trial in TRIALS:
        _write(tmp_path, "current-11", trial, progress=1.1, limits=8, kills=100, items=20)
        _write(
            tmp_path,
            "map-navigation-prior-v1",
            trial,
            progress=1.3,
            limits=3,
            kills=85,
            items=18,
        )

    comparison = compare_map_navigation(tmp_path)

    assert comparison["passed"] is True
    assert comparison["decision"] == "authorize_navigation_training_comparison"


def test_map_navigation_comparison_rejects_incomplete_evidence(tmp_path: Path) -> None:
    _write(tmp_path, "current-11", "deterministic", progress=1.1, limits=8, kills=100, items=20)

    comparison = compare_map_navigation(tmp_path)

    assert comparison["passed"] is False
    assert comparison["decision"] == "invalid_or_incomplete_evidence"
