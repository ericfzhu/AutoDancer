"""Aggregate frozen-policy curriculum feasibility trials."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRIALS = ("deterministic", "stochastic-95001", "stochastic-95002")


def _successful(result: dict[str, Any], target_level: int) -> bool:
    zone = int(result.get("furthest_zone", 0))
    floor = int(result.get("furthest_floor", 0))
    progress = (zone - 1) * 4 + floor if zone > 0 and floor > 0 else 0
    return str(result.get("status")) == "curriculum_complete" and progress >= target_level


def compare_curriculum_feasibility(root: Path) -> dict[str, Any]:
    evaluation = root / "evaluation"
    target_level = 5
    trials: dict[str, Any] = {}
    successful_seeds: set[int] = set()
    successful_modes = 0
    for name in TRIALS:
        path = evaluation / name / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("controller_valid"):
            raise ValueError(f"Controller-invalid curriculum report: {path}")
        if int(report.get("curriculum_start_level", 0)) != 4:
            raise ValueError(f"Unexpected curriculum start level in {path}")
        if int(report.get("curriculum_target_level", 0)) != target_level:
            raise ValueError(f"Unexpected curriculum target level in {path}")
        results = list(report["trained"]["results"])
        successes = [result for result in results if _successful(result, target_level)]
        seeds = sorted(int(result["seed"]) for result in successes)
        successful_seeds.update(seeds)
        successful_modes += int(bool(seeds))
        trials[name] = {
            "episodes": len(results),
            "successes": len(successes),
            "completion_rate": len(successes) / max(len(results), 1),
            "successful_seeds": seeds,
            "deaths": sum(result.get("status") == "dead" for result in results),
            "step_limits": sum(result.get("status") == "step_limit" for result in results),
            "mean_turns": sum(int(result.get("turns", 0)) for result in results)
            / max(len(results), 1),
            "enemy_kills": sum(int(result.get("enemy_kills", 0)) for result in results),
            "player_damage": sum(int(result.get("player_damage", 0)) for result in results),
        }
    supported = len(successful_seeds) >= 3 and successful_modes >= 2
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0013",
        "created_at": datetime.now(UTC).isoformat(),
        "curriculum_start_level": 4,
        "curriculum_target_level": target_level,
        "trials": trials,
        "distinct_successful_seeds": sorted(successful_seeds),
        "successful_policy_modes": successful_modes,
        "existing_boss_competence_supported": supported,
        "decision": (
            "proceed_to_backward_curriculum"
            if supported
            else "train_boss_to_zone2_curriculum"
        ),
        "normal_start_promotable": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ".comparison.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(root / "comparison.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EXP-0013 curriculum trials")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(compare_curriculum_feasibility(arguments.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
