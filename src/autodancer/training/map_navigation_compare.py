"""Compare the predeclared EXP-0012 hybrid map-navigation ablation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARMS = ("current-11", "map-navigation-prior-v1")
TRIALS = ("deterministic", "stochastic-94001", "stochastic-94002")
EXPECTED_SEEDS = tuple(range(62_001, 62_025))
EXPECTED_CONTRACTS = {
    "current-11": "current",
    "map-navigation-prior-v1": "map-navigation-prior-v1",
}


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _valid(report: dict[str, Any] | None, arm: str, trial: str) -> bool:
    if report is None:
        return False
    mode = "deterministic" if trial == "deterministic" else "stochastic"
    policy_seed = 0 if mode == "deterministic" else int(trial.rsplit("-", 1)[1])
    results = report.get("trained", {}).get("results", [])
    return bool(
        report.get("controller_valid") is True
        and int(report.get("worker_restarts", 0)) == 0
        and not report.get("infrastructure_events")
        and report.get("policy_mode") == mode
        and int(report.get("policy_seed", -1)) == policy_seed
        and report.get("action_contract") == EXPECTED_CONTRACTS[arm]
        and tuple(int(seed) for seed in report.get("seeds", [])) == EXPECTED_SEEDS
        and tuple(int(result["seed"]) for result in results) == EXPECTED_SEEDS
    )


def _summary(report: dict[str, Any] | None, arm: str, trial: str) -> dict[str, Any]:
    trained = report.get("trained", {}) if report else {}
    results = list(trained.get("results", []))
    zone_two = {
        int(result["seed"])
        for result in results
        if int(result.get("furthest_zone", 0)) >= 2
    }
    floor_two = {
        int(result["seed"])
        for result in results
        if int(result.get("furthest_zone", 0)) >= 2
        or int(result.get("furthest_floor", 0)) >= 2
    }
    return {
        "valid": _valid(report, arm, trial),
        "episodes": len(results),
        "turns": sum(int(result.get("turns", 0)) for result in results),
        "mean_progress": float(trained.get("mean_progress", 0.0)),
        "furthest_zone": int(trained.get("furthest_zone", 0)),
        "furthest_floor": int(trained.get("furthest_floor", 0)),
        "floor_two_seeds": sorted(floor_two),
        "zone_two_seeds": sorted(zone_two),
        "step_limits": sum(result.get("status") == "step_limit" for result in results),
        "enemy_kills": int(trained.get("enemy_kills", 0)),
        "item_pickups": int(trained.get("item_pickups", 0)),
        "unique_positions_per_100_turns": float(
            trained.get("unique_positions_per_100_turns", 0.0)
        ),
        "staircase_discoveries": int(trained.get("staircase_discoveries", 0)),
        "staircase_exits": int(trained.get("staircase_exits", 0)),
        "navigation_prior_rate": float(trained.get("navigation_prior_rate", 0.0)),
        "mean_navigation_masked_directions": float(
            trained.get("mean_navigation_masked_directions", 0.0)
        ),
        "action_outcome_counts": dict(trained.get("action_outcome_counts", {})),
    }


def _aggregate(trials: dict[str, dict[str, Any]]) -> dict[str, Any]:
    episodes = sum(item["episodes"] for item in trials.values())
    return {
        "episodes": episodes,
        "turns": sum(item["turns"] for item in trials.values()),
        "mean_progress": sum(
            item["mean_progress"] * item["episodes"] for item in trials.values()
        )
        / max(episodes, 1),
        "furthest_zone": max(item["furthest_zone"] for item in trials.values()),
        "furthest_floor": max(item["furthest_floor"] for item in trials.values()),
        "floor_two_seeds": sorted(
            {seed for item in trials.values() for seed in item["floor_two_seeds"]}
        ),
        "zone_two_seeds": sorted(
            {seed for item in trials.values() for seed in item["zone_two_seeds"]}
        ),
        "step_limits": sum(item["step_limits"] for item in trials.values()),
        "step_limit_rate": sum(item["step_limits"] for item in trials.values())
        / max(episodes, 1),
        "enemy_kills": sum(item["enemy_kills"] for item in trials.values()),
        "item_pickups": sum(item["item_pickups"] for item in trials.values()),
        "staircase_discoveries": sum(
            item["staircase_discoveries"] for item in trials.values()
        ),
        "staircase_exits": sum(item["staircase_exits"] for item in trials.values()),
    }


def compare_map_navigation(root: Path) -> dict[str, Any]:
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
    guided = aggregates["map-navigation-prior-v1"]
    valid = all(summaries[arm][trial]["valid"] for arm in ARMS for trial in TRIALS)
    improved_trials = [
        trial
        for trial in TRIALS
        if summaries["map-navigation-prior-v1"][trial]["mean_progress"]
        > summaries["current-11"][trial]["mean_progress"]
    ]
    sampled_zone_two_sets = [
        set(summaries["map-navigation-prior-v1"][trial]["zone_two_seeds"])
        for trial in TRIALS[1:]
    ]
    repeatable_zone_two = sorted(set.intersection(*sampled_zone_two_sets))
    criteria = {
        "all_controller_reports_valid": valid,
        "mean_floor_progress_improved": guided["mean_progress"] > current["mean_progress"],
        "step_limit_rate_halved": guided["step_limit_rate"]
        <= 0.5 * current["step_limit_rate"],
        "kills_retained_at_80_percent": guided["enemy_kills"]
        >= 0.8 * current["enemy_kills"],
        "items_retained_at_80_percent": guided["item_pickups"]
        >= 0.8 * current["item_pickups"],
        "progress_improved_in_two_modes": len(improved_trials) >= 2,
    }
    passed = all(criteria.values())
    zone_two_promotable = bool(
        len(guided["zone_two_seeds"]) >= 3 and len(repeatable_zone_two) >= 2
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": "EXP-0012",
        "arms": {
            arm: {"trials": summaries[arm], "aggregate": aggregates[arm]}
            for arm in ARMS
        },
        "progress_improved_trials": improved_trials,
        "repeatable_stochastic_zone_two_seeds": repeatable_zone_two,
        "criteria": criteria,
        "passed": passed,
        "zone_two_promotable": zone_two_promotable,
        "decision": (
            "zone2_promotion_candidate"
            if passed and zone_two_promotable
            else "authorize_navigation_training_comparison"
            if passed
            else "retain_current_action_contract"
            if valid
            else "invalid_or_incomplete_evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EXP-0012 navigation contracts")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = compare_map_navigation(arguments.root)
    output = arguments.output or arguments.root / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"decision": report["decision"], "output": str(output)}, sort_keys=True))
    return 1 if report["decision"] == "invalid_or_incomplete_evidence" else 0


if __name__ == "__main__":
    raise SystemExit(main())
