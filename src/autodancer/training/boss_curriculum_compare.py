"""Compare EXP-0014 boss-curriculum architecture arms."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARMS = ("a2-boss-curriculum", "a8-boss-curriculum")
TRIALS = (
    ("deterministic", "deterministic", 0),
    ("stochastic-96001", "stochastic", 96001),
    ("stochastic-96002", "stochastic", 96002),
)
TARGET_LEVEL = 5


def _progress(result: dict[str, Any]) -> int:
    zone = int(result.get("furthest_zone", 0))
    floor = int(result.get("furthest_floor", 0))
    return (zone - 1) * 4 + floor if zone > 0 and floor > 0 else 0


def _success(result: dict[str, Any]) -> bool:
    return (
        str(result.get("status")) == "curriculum_complete"
        and _progress(result) >= TARGET_LEVEL
    )


def _training_summary(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Training metrics are empty: {path}")
    final_step = max(int(record.get("global_step", 0)) for record in records)
    first_cut = final_step / 4
    last_cut = final_step * 3 / 4

    def window(predicate: Any) -> tuple[float, float]:
        selected = [record for record in records if predicate(int(record["global_step"]))]
        episodes = sum(float(record.get("episodes", 0.0)) for record in selected)
        completions = sum(
            float(record.get("curriculum_completions", 0.0)) for record in selected
        )
        return completions, completions / max(episodes, 1.0)

    first_completions, first_rate = window(lambda step: step <= first_cut)
    last_completions, last_rate = window(lambda step: step > last_cut)
    seeds = {
        int(seed)
        for record in records
        for seed in record.get("episode_seeds", [])
    }
    finite_losses = all(
        math.isfinite(float(record[name]))
        for record in records
        for name in ("policy_loss", "value_loss", "entropy")
        if name in record
    )
    return {
        "final_step": final_step,
        "updates": max(int(record.get("updates", 0)) for record in records),
        "episodes": sum(float(record.get("episodes", 0.0)) for record in records),
        "curriculum_completions": sum(
            float(record.get("curriculum_completions", 0.0)) for record in records
        ),
        "time_limits": sum(float(record.get("time_limits", 0.0)) for record in records),
        "distinct_episode_seeds": sorted(seeds),
        "first_quarter_completions": first_completions,
        "first_quarter_completion_rate": first_rate,
        "last_quarter_completions": last_completions,
        "last_quarter_completion_rate": last_rate,
        "finite_losses": finite_losses,
        "worker_restarts": max(
            int(record.get("worker_restarts", 0)) for record in records
        ),
    }


def compare_boss_curriculum(root: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    passing: list[str] = []
    extendable: list[str] = []
    for arm in ARMS:
        trials: dict[str, Any] = {}
        successful_seeds: set[int] = set()
        successful_modes = 0
        total_successes = 0
        total_episodes = 0
        total_deaths = 0
        success_turns: list[int] = []
        controller_valid = True
        for trial, expected_mode, expected_policy_seed in TRIALS:
            path = root / "evaluation" / arm / trial / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            valid = bool(report.get("controller_valid")) and not report.get(
                "infrastructure_events"
            )
            valid = valid and int(report.get("worker_restarts", 0)) == 0
            valid = valid and report.get("policy_mode") == expected_mode
            valid = valid and int(report.get("policy_seed", -1)) == expected_policy_seed
            valid = valid and int(report.get("curriculum_start_level", 0)) == 4
            valid = valid and int(report.get("curriculum_target_level", 0)) == TARGET_LEVEL
            controller_valid = controller_valid and valid
            results = list(report["trained"]["results"])
            successes = [result for result in results if _success(result)]
            seeds = sorted(int(result["seed"]) for result in successes)
            successful_seeds.update(seeds)
            successful_modes += int(bool(successes))
            total_successes += len(successes)
            total_episodes += len(results)
            total_deaths += sum(str(result.get("status")) == "dead" for result in results)
            success_turns.extend(int(result.get("turns", 0)) for result in successes)
            trials[trial] = {
                "valid": valid,
                "episodes": len(results),
                "successes": len(successes),
                "completion_rate": len(successes) / max(len(results), 1),
                "successful_seeds": seeds,
                "deaths": sum(str(result.get("status")) == "dead" for result in results),
                "step_limits": sum(
                    str(result.get("status")) == "step_limit" for result in results
                ),
            }
        training = _training_summary(root / "training" / arm / "metrics.jsonl")
        passed = (
            controller_valid
            and training["finite_losses"]
            and training["worker_restarts"] == 0
            and len(successful_seeds) >= 3
            and successful_modes >= 2
        )
        can_extend = (
            not passed
            and controller_valid
            and bool(successful_seeds)
            and training["last_quarter_completion_rate"]
            > training["first_quarter_completion_rate"]
        )
        passing.extend([arm] if passed else [])
        extendable.extend([arm] if can_extend else [])
        arms[arm] = {
            "training": training,
            "trials": trials,
            "controller_valid": controller_valid,
            "distinct_successful_seeds": sorted(successful_seeds),
            "successful_policy_modes": successful_modes,
            "aggregate_completion_rate": total_successes / max(total_episodes, 1),
            "death_rate": total_deaths / max(total_episodes, 1),
            "mean_success_turns": (
                sum(success_turns) / len(success_turns) if success_turns else None
            ),
            "passed": passed,
            "extendable": can_extend,
        }

    def rank(arm: str) -> tuple[float, ...]:
        value = arms[arm]
        mean_turns = value["mean_success_turns"]
        return (
            float(len(value["distinct_successful_seeds"])),
            float(value["aggregate_completion_rate"]),
            -float(value["death_rate"]),
            -float(mean_turns if mean_turns is not None else math.inf),
        )

    selected = max(passing, key=rank) if passing else None
    extension = max(extendable, key=rank) if extendable and selected is None else None
    decision = (
        "advance_to_floor3_curriculum"
        if selected is not None
        else "extend_learning_arm"
        if extension is not None
        else "declare_boss_tactical_action_intervention"
    )
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0014",
        "created_at": datetime.now(UTC).isoformat(),
        "arms": arms,
        "passing_arms": passing,
        "selected_arm": selected,
        "extension_arm": extension,
        "decision": decision,
        "normal_start_promotable": False,
    }
    temporary = root / ".comparison.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(root / "comparison.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EXP-0014 boss curricula")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(compare_boss_curriculum(arguments.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
