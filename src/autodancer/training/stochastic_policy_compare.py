"""Compare the predeclared EXP-0009 policy-execution trials."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARMS = ("a2-frozen", "a2-continuation", "a8-continuation")
TRIALS = ("deterministic", "stochastic-91001", "stochastic-91002")
STOCHASTIC_TRIALS = TRIALS[1:]
EXPECTED_SEEDS = tuple(range(57_001, 57_025))


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _valid(report: dict[str, Any] | None, trial: str) -> bool:
    if report is None:
        return False
    expected_mode = "deterministic" if trial == "deterministic" else "stochastic"
    expected_policy_seed = 0 if trial == "deterministic" else int(trial.rsplit("-", 1)[1])
    results = report.get("trained", {}).get("results", [])
    return bool(
        report.get("controller_valid") is True
        and int(report.get("worker_restarts", 0)) == 0
        and not report.get("infrastructure_events")
        and report.get("policy_mode") == expected_mode
        and int(report.get("policy_seed", -1)) == expected_policy_seed
        and tuple(int(seed) for seed in report.get("seeds", [])) == EXPECTED_SEEDS
        and tuple(int(result["seed"]) for result in results) == EXPECTED_SEEDS
    )


def _zone_two_seeds(report: dict[str, Any] | None) -> set[int]:
    if report is None:
        return set()
    return {
        int(result["seed"])
        for result in report.get("trained", {}).get("results", [])
        if int(result.get("furthest_zone", 0)) >= 2
    }


def _trial_summary(report: dict[str, Any] | None, trial: str) -> dict[str, Any]:
    zone_two = sorted(_zone_two_seeds(report))
    trained = report.get("trained", {}) if report else {}
    return {
        "valid": _valid(report, trial),
        "zone_two_seeds": zone_two,
        "zone_two_count": len(zone_two),
        "furthest_zone": int(trained.get("furthest_zone", 0)),
        "furthest_floor": int(trained.get("furthest_floor", 0)),
        "mean_progress": float(trained.get("mean_progress", 0.0)),
        "death_rate": float(trained.get("death_rate", 0.0)),
        "step_limit_rate": float(trained.get("step_limit_rate", 0.0)),
        "enemy_kills": int(trained.get("enemy_kills", 0)),
        "item_pickups": int(trained.get("item_pickups", 0)),
    }


def compare_stochastic_policy(root: Path) -> dict[str, Any]:
    reports = {
        arm: {
            trial: _read(root / "evaluation" / arm / trial / "report.json")
            for trial in TRIALS
        }
        for arm in ARMS
    }
    arms: dict[str, Any] = {}
    for arm, arm_reports in reports.items():
        trial_summaries = {
            trial: _trial_summary(report, trial) for trial, report in arm_reports.items()
        }
        stochastic_sets = [
            _zone_two_seeds(arm_reports[trial]) for trial in STOCHASTIC_TRIALS
        ]
        stochastic_union = set().union(*stochastic_sets)
        stochastic_intersection = set.intersection(*stochastic_sets)
        arms[arm] = {
            "trials": trial_summaries,
            "distinct_stochastic_zone_two_seeds": sorted(stochastic_union),
            "repeatable_stochastic_zone_two_seeds": sorted(stochastic_intersection),
            "passes": (
                len(stochastic_union) >= 3 and len(stochastic_intersection) >= 2
            ),
        }

    complete = all(
        reports[arm][trial] is not None for arm in ARMS for trial in TRIALS
    )
    valid = complete and all(
        arms[arm]["trials"][trial]["valid"] for arm in ARMS for trial in TRIALS
    )
    passing_arms = [arm for arm in ARMS if arms[arm]["passes"]]
    any_stochastic_zone_two = any(
        arms[arm]["distinct_stochastic_zone_two_seeds"] for arm in ARMS
    )
    if not valid:
        decision = "invalid_or_incomplete_evidence"
    elif passing_arms:
        decision = "stochastic_execution_reaches_zone2_repeatably"
    elif any_stochastic_zone_two:
        decision = "isolated_stochastic_zone2_inconclusive"
    else:
        decision = "reject_execution_mode_mismatch"
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": "EXP-0009",
        "evaluation_seeds": list(EXPECTED_SEEDS),
        "policy_seeds": [0, 91_001, 91_002],
        "arms": arms,
        "criteria": {
            "complete": complete,
            "all_controller_reports_valid": valid,
            "three_distinct_zone_two_game_seeds": any(
                len(arms[arm]["distinct_stochastic_zone_two_seeds"]) >= 3
                for arm in ARMS
            ),
            "two_zone_two_seeds_repeat_across_policy_samples": any(
                len(arms[arm]["repeatable_stochastic_zone_two_seeds"]) >= 2
                for arm in ARMS
            ),
        },
        "passing_arms": passing_arms,
        "passed": bool(valid and passing_arms),
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EXP-0009 execution modes")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = compare_stochastic_policy(arguments.root)
    output = arguments.output or arguments.root / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"decision": report["decision"], "output": str(output)}, sort_keys=True))
    return 0 if report["decision"] != "invalid_or_incomplete_evidence" else 1


if __name__ == "__main__":
    raise SystemExit(main())
