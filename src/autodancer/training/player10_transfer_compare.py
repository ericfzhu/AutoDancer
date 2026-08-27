"""Decide the predeclared EXP-0017 frozen player10 transfer gate."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODES = (
    ("deterministic", "deterministic", 0),
    ("stochastic-100001", "stochastic", 100001),
    ("stochastic-100002", "stochastic", 100002),
)


def _success(result: dict[str, Any]) -> bool:
    return (
        str(result.get("status")) == "curriculum_complete"
        and int(result.get("furthest_zone", 0)) >= 2
    )


def compare_player10_transfer(root: Path) -> dict[str, Any]:
    selection = json.loads((root / "heldout-selection.json").read_text(encoding="utf-8-sig"))
    expected_seeds = [int(seed) for seed in selection["seeds"]]
    if len(expected_seeds) != 24 or len(set(expected_seeds)) != 24:
        raise ValueError("EXP-0017 transfer gate requires exactly 24 distinct seeds")
    reports: dict[str, Any] = {}
    sampled_successes = 0
    sampled_episodes = 0
    sampled_deaths = 0
    sampled_successful_seeds: set[int] = set()
    controller_valid = True
    deterministic_rate = 0.0
    warnings: list[dict[str, Any]] = []
    for name, expected_mode, expected_policy_seed in MODES:
        report = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        results = list(report["trained"]["results"])
        result_seeds = [int(result["seed"]) for result in results]
        valid = bool(report.get("controller_valid"))
        valid = valid and not report.get("infrastructure_events")
        valid = valid and int(report.get("worker_restarts", 0)) == 0
        valid = valid and report.get("policy_mode") == expected_mode
        valid = valid and int(report.get("policy_seed", -1)) == expected_policy_seed
        valid = valid and int(report.get("curriculum_start_level", 0)) == 4
        valid = valid and int(report.get("curriculum_target_level", 0)) == 5
        valid = valid and report.get("curriculum_profile") == "boss1hp-player10"
        valid = valid and result_seeds == expected_seeds
        successes = [result for result in results if _success(result)]
        deaths = sum(str(result.get("status")) == "dead" for result in results)
        successful_seeds = sorted(int(result["seed"]) for result in successes)
        completion_rate = len(successes) / max(len(results), 1)
        if expected_mode == "deterministic":
            deterministic_rate = completion_rate
        else:
            sampled_successes += len(successes)
            sampled_episodes += len(results)
            sampled_deaths += deaths
            sampled_successful_seeds.update(successful_seeds)
        for result in results:
            if int(result.get("boss_kills", 0)) > 0 and int(result.get("boss_damage", 0)) == 0:
                warnings.append(
                    {
                        "mode": name,
                        "seed": int(result["seed"]),
                        "status": result.get("status"),
                        "kind": "boss_kill_without_boss_damage",
                    }
                )
        reports[name] = {
            "valid": valid,
            "episodes": len(results),
            "successes": len(successes),
            "completion_rate": completion_rate,
            "deaths": deaths,
            "successful_seeds": successful_seeds,
        }
        controller_valid = controller_valid and valid

    sampled_completion_rate = sampled_successes / max(sampled_episodes, 1)
    sampled_death_rate = sampled_deaths / max(sampled_episodes, 1)
    passed = (
        controller_valid
        and sampled_completion_rate >= 0.5
        and len(sampled_successful_seeds) >= 12
        and sampled_death_rate <= 0.5
        and deterministic_rate >= 0.2
    )
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0017",
        "created_at": datetime.now(UTC).isoformat(),
        "reports": reports,
        "sampled_aggregate": {
            "episodes": sampled_episodes,
            "successes": sampled_successes,
            "completion_rate": sampled_completion_rate,
            "deaths": sampled_deaths,
            "death_rate": sampled_death_rate,
            "distinct_successful_seeds": sorted(sampled_successful_seeds),
        },
        "deterministic_completion_rate": deterministic_rate,
        "controller_valid": controller_valid,
        "passed": passed,
        "decision": (
            "retain_parent_and_advance_to_player6" if passed else "run_mixed_player10_replay"
        ),
        "selected_checkpoint": (
            "runs/assisted-death-metal/training/seed-68002/final.pt" if passed else None
        ),
        "normal_start_promotable": False,
        "diagnostic_warnings": warnings,
    }
    temporary = root / ".comparison.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(root / "comparison.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide EXP-0017 frozen transfer gate")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(compare_player10_transfer(arguments.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
