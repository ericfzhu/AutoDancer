from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.stochastic_policy_compare import (
    ARMS,
    EXPECTED_SEEDS,
    TRIALS,
    compare_stochastic_policy,
)


def _write_report(path: Path, trial: str, zone_two: set[int]) -> None:
    mode = "deterministic" if trial == "deterministic" else "stochastic"
    policy_seed = 0 if mode == "deterministic" else int(trial.rsplit("-", 1)[1])
    results = [
        {
            "seed": seed,
            "furthest_zone": 2 if seed in zone_two else 1,
            "furthest_floor": 1,
        }
        for seed in EXPECTED_SEEDS
    ]
    report = {
        "controller_valid": True,
        "worker_restarts": 0,
        "infrastructure_events": [],
        "policy_mode": mode,
        "policy_seed": policy_seed,
        "seeds": list(EXPECTED_SEEDS),
        "trained": {
            "results": results,
            "furthest_zone": max(result["furthest_zone"] for result in results),
            "furthest_floor": 1,
            "mean_progress": 1.0,
            "death_rate": 0.0,
            "step_limit_rate": 0.0,
            "enemy_kills": 0,
            "item_pickups": 0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def _complete_fixture(root: Path) -> None:
    for arm in ARMS:
        for trial in TRIALS:
            _write_report(root / "evaluation" / arm / trial / "report.json", trial, set())


def test_comparison_passes_only_repeatable_multiseed_zone_two(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    first = {57_001, 57_002, 57_003}
    second = {57_001, 57_002, 57_004}
    _write_report(
        tmp_path / "evaluation" / "a2-frozen" / "stochastic-91001" / "report.json",
        "stochastic-91001",
        first,
    )
    _write_report(
        tmp_path / "evaluation" / "a2-frozen" / "stochastic-91002" / "report.json",
        "stochastic-91002",
        second,
    )

    result = compare_stochastic_policy(tmp_path)

    assert result["passed"] is True
    assert result["passing_arms"] == ["a2-frozen"]
    assert result["arms"]["a2-frozen"]["repeatable_stochastic_zone_two_seeds"] == [
        57_001,
        57_002,
    ]


def test_comparison_calls_one_zone_two_trajectory_inconclusive(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    _write_report(
        tmp_path / "evaluation" / "a8-continuation" / "stochastic-91001" / "report.json",
        "stochastic-91001",
        {57_001},
    )

    result = compare_stochastic_policy(tmp_path)

    assert result["passed"] is False
    assert result["decision"] == "isolated_stochastic_zone2_inconclusive"


def test_comparison_rejects_wrong_policy_seed(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "evaluation" / "a2-frozen" / "stochastic-91001" / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["policy_seed"] = 123
    path.write_text(json.dumps(report), encoding="utf-8")

    result = compare_stochastic_policy(tmp_path)

    assert result["decision"] == "invalid_or_incomplete_evidence"
    assert result["criteria"]["all_controller_reports_valid"] is False
