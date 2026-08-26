"""Compare the predeclared EXP-0011 action-contract inference ablation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARMS = ("current-11", "known-invalid-wall-v1")
TRIALS = ("deterministic", "stochastic-93001", "stochastic-93002")
EXPECTED_SEEDS = tuple(range(61_001, 61_025))
EXPECTED_CONTRACTS = {
    "current-11": "current",
    "known-invalid-wall-v1": "known-invalid-wall-v1",
}


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _valid(report: dict[str, Any] | None, arm: str, trial: str) -> bool:
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
        and report.get("action_contract") == EXPECTED_CONTRACTS[arm]
        and tuple(int(seed) for seed in report.get("seeds", [])) == EXPECTED_SEEDS
        and tuple(int(result["seed"]) for result in results) == EXPECTED_SEEDS
    )


def _summary(report: dict[str, Any] | None, arm: str, trial: str) -> dict[str, Any]:
    trained = report.get("trained", {}) if report else {}
    outcomes = dict(trained.get("action_outcome_counts", {}))
    results = list(trained.get("results", []))
    turns = sum(int(result.get("turns", 0)) for result in results)
    step_limits = sum(result.get("status") == "step_limit" for result in results)
    zone_two_seeds = sorted(
        int(result["seed"])
        for result in results
        if int(result.get("furthest_zone", 0)) >= 2
    )
    return {
        "valid": _valid(report, arm, trial),
        "episodes": len(results),
        "turns": turns,
        "wall_attempts": int(outcomes.get("wall_attempt", 0)),
        "wall_attempt_rate": int(outcomes.get("wall_attempt", 0)) / max(turns, 1),
        "step_limits": step_limits,
        "step_limit_rate": step_limits / max(len(results), 1),
        "mean_progress": float(trained.get("mean_progress", 0.0)),
        "furthest_zone": int(trained.get("furthest_zone", 0)),
        "furthest_floor": int(trained.get("furthest_floor", 0)),
        "zone_two_seeds": zone_two_seeds,
        "enemy_kills": int(trained.get("enemy_kills", 0)),
        "item_pickups": int(trained.get("item_pickups", 0)),
        "known_invalid_wall_discoveries": int(
            trained.get("known_invalid_wall_discoveries", 0)
        ),
        "mean_masked_directions": float(trained.get("mean_masked_directions", 0.0)),
    }


def _aggregate(trials: dict[str, dict[str, Any]]) -> dict[str, Any]:
    episodes = sum(item["episodes"] for item in trials.values())
    turns = sum(item["turns"] for item in trials.values())
    wall_attempts = sum(item["wall_attempts"] for item in trials.values())
    step_limits = sum(item["step_limits"] for item in trials.values())
    return {
        "episodes": episodes,
        "turns": turns,
        "wall_attempts": wall_attempts,
        "wall_attempt_rate": wall_attempts / max(turns, 1),
        "step_limits": step_limits,
        "step_limit_rate": step_limits / max(episodes, 1),
        "mean_progress": sum(
            item["mean_progress"] * item["episodes"] for item in trials.values()
        )
        / max(episodes, 1),
        "furthest_zone": max(item["furthest_zone"] for item in trials.values()),
        "furthest_floor": max(item["furthest_floor"] for item in trials.values()),
        "zone_two_seeds": sorted(
            {seed for item in trials.values() for seed in item["zone_two_seeds"]}
        ),
        "enemy_kills": sum(item["enemy_kills"] for item in trials.values()),
        "item_pickups": sum(item["item_pickups"] for item in trials.values()),
        "known_invalid_wall_discoveries": sum(
            item["known_invalid_wall_discoveries"] for item in trials.values()
        ),
    }


def compare_action_contract(root: Path) -> dict[str, Any]:
    summaries = {
        arm: {
            trial: _summary(
                _read(root / "evaluation" / f"{arm}-{trial}" / "report.json"),
                arm,
                trial,
            )
            for trial in TRIALS
        }
        for arm in ARMS
    }
    aggregates = {arm: _aggregate(summaries[arm]) for arm in ARMS}
    current = aggregates["current-11"]
    masked = aggregates["known-invalid-wall-v1"]
    valid = all(summaries[arm][trial]["valid"] for arm in ARMS for trial in TRIALS)
    wall_reduction = (
        1.0 - masked["wall_attempt_rate"] / current["wall_attempt_rate"]
        if current["wall_attempt_rate"] > 0
        else 0.0
    )
    efficient_trials = [
        trial
        for trial in TRIALS
        if summaries["known-invalid-wall-v1"][trial]["wall_attempt_rate"]
        < summaries["current-11"][trial]["wall_attempt_rate"]
    ]
    criteria = {
        "all_controller_reports_valid": valid,
        "wall_attempt_rate_reduced_by_75_percent": wall_reduction >= 0.75,
        "step_limit_rate_reduced": masked["step_limit_rate"] < current["step_limit_rate"],
        "mean_floor_progress_not_reduced": masked["mean_progress"] >= current["mean_progress"],
        "kills_retained_at_80_percent": masked["enemy_kills"] >= 0.8 * current["enemy_kills"],
        "items_retained_at_80_percent": masked["item_pickups"] >= 0.8 * current["item_pickups"],
        "action_efficiency_improved_in_two_modes": len(efficient_trials) >= 2,
    }
    passed = all(criteria.values())
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": "EXP-0011",
        "arms": {
            arm: {"trials": summaries[arm], "aggregate": aggregates[arm]}
            for arm in ARMS
        },
        "wall_attempt_rate_reduction": wall_reduction,
        "action_efficiency_improved_trials": efficient_trials,
        "criteria": criteria,
        "passed": passed,
        "decision": (
            "authorize_controlled_training_comparison"
            if passed
            else "retain_current_action_contract"
            if valid
            else "invalid_or_incomplete_evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EXP-0011 action contracts")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = compare_action_contract(arguments.root)
    output = arguments.output or arguments.root / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"decision": report["decision"], "output": str(output)}, sort_keys=True))
    return 1 if report["decision"] == "invalid_or_incomplete_evidence" else 0


if __name__ == "__main__":
    raise SystemExit(main())
