"""Aggregate and decide the three-trial assisted boss acquisition experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRIAL_SEEDS = (68001, 68002, 68003)
MODES = (
    ("deterministic", "deterministic", 0),
    ("stochastic-98001", "stochastic", 98001),
    ("stochastic-98002", "stochastic", 98002),
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


def _training_valid(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"training metrics are empty: {path}")
    finite = all(
        math.isfinite(float(record[name]))
        for record in records
        for name in ("policy_loss", "value_loss", "entropy")
        if name in record
    )
    return {
        "final_step": max(int(record.get("global_step", 0)) for record in records),
        "updates": max(int(record.get("updates", 0)) for record in records),
        "finite_losses": finite,
        "worker_restarts": max(int(record.get("worker_restarts", 0)) for record in records),
        "episodes": int(sum(float(record.get("episodes", 0)) for record in records)),
        "curriculum_completions": int(
            sum(float(record.get("curriculum_completions", 0)) for record in records)
        ),
    }


def compare_assisted_boss(root: Path) -> dict[str, Any]:
    trials: dict[str, Any] = {}
    all_successful_seeds: set[int] = set()
    successful_training_trials = 0
    aggregate_successes = 0
    aggregate_episodes = 0
    natural_restarts = 0
    controller_valid = True
    attribution_warnings: list[dict[str, Any]] = []
    for training_seed in TRIAL_SEEDS:
        trial_successful_seeds: set[int] = set()
        trial_successes = 0
        trial_episodes = 0
        trial_deaths = 0
        success_turns: list[int] = []
        reports: dict[str, Any] = {}
        trial_controller_valid = True
        for directory, expected_mode, expected_policy_seed in MODES:
            path = root / "evaluation" / f"seed-{training_seed}" / directory / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            valid = bool(report.get("controller_valid"))
            valid = valid and not report.get("infrastructure_events")
            valid = valid and int(report.get("worker_restarts", 0)) == 0
            valid = valid and report.get("policy_mode") == expected_mode
            valid = valid and int(report.get("policy_seed", -1)) == expected_policy_seed
            valid = valid and int(report.get("curriculum_start_level", 0)) == 4
            valid = valid and int(report.get("curriculum_target_level", 0)) == TARGET_LEVEL
            valid = valid and report.get("curriculum_profile") == "boss1hp-player20"
            results = list(report["trained"]["results"])
            successes = [result for result in results if _success(result)]
            seeds = sorted(int(result["seed"]) for result in successes)
            trial_successful_seeds.update(seeds)
            trial_successes += len(successes)
            trial_episodes += len(results)
            trial_deaths += sum(str(result.get("status")) == "dead" for result in results)
            success_turns.extend(int(result.get("turns", 0)) for result in successes)
            natural_restarts += int(report.get("worker_restarts", 0))
            for result in results:
                if int(result.get("boss_kills", 0)) > 0 and int(result.get("boss_damage", 0)) == 0:
                    attribution_warnings.append(
                        {
                            "training_seed": training_seed,
                            "mode": directory,
                            "game_seed": int(result["seed"]),
                            "status": result.get("status"),
                            "boss_kills": int(result.get("boss_kills", 0)),
                            "boss_damage": int(result.get("boss_damage", 0)),
                        }
                    )
            reports[directory] = {
                "valid": valid,
                "episodes": len(results),
                "successes": len(successes),
                "completion_rate": len(successes) / max(len(results), 1),
                "deaths": sum(str(result.get("status")) == "dead" for result in results),
                "successful_seeds": seeds,
            }
            trial_controller_valid = trial_controller_valid and valid
        training = _training_valid(
            root / "training" / f"seed-{training_seed}" / "metrics.jsonl"
        )
        trial_controller_valid = (
            trial_controller_valid
            and training["finite_losses"]
            and training["worker_restarts"] == 0
            and training["final_step"] == 51200
        )
        successful_training_trials += int(bool(trial_successful_seeds))
        all_successful_seeds.update(trial_successful_seeds)
        aggregate_successes += trial_successes
        aggregate_episodes += trial_episodes
        controller_valid = controller_valid and trial_controller_valid
        trials[str(training_seed)] = {
            "training": training,
            "reports": reports,
            "controller_valid": trial_controller_valid,
            "distinct_successful_seeds": sorted(trial_successful_seeds),
            "aggregate_successes": trial_successes,
            "aggregate_episodes": trial_episodes,
            "aggregate_completion_rate": trial_successes / max(trial_episodes, 1),
            "death_rate": trial_deaths / max(trial_episodes, 1),
            "median_success_turns": (
                statistics.median(success_turns) if success_turns else None
            ),
        }

    aggregate_completion_rate = aggregate_successes / max(aggregate_episodes, 1)
    passed = (
        controller_valid
        and natural_restarts == 0
        and len(all_successful_seeds) >= 3
        and successful_training_trials >= 2
        and aggregate_completion_rate >= 0.2
    )

    def rank(training_seed: int) -> tuple[float, ...]:
        trial = trials[str(training_seed)]
        median_turns = trial["median_success_turns"]
        return (
            float(len(trial["distinct_successful_seeds"])),
            float(trial["aggregate_completion_rate"]),
            -float(trial["death_rate"]),
            -float(median_turns if median_turns is not None else math.inf),
        )

    selected_seed = max(TRIAL_SEEDS, key=rank) if passed else None
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0016",
        "created_at": datetime.now(UTC).isoformat(),
        "trials": trials,
        "aggregate": {
            "episodes": aggregate_episodes,
            "successes": aggregate_successes,
            "completion_rate": aggregate_completion_rate,
            "distinct_successful_seeds": sorted(all_successful_seeds),
            "successful_training_trials": successful_training_trials,
            "controller_valid": controller_valid,
            "natural_restarts": natural_restarts,
        },
        "passed": passed,
        "selected_training_seed": selected_seed,
        "selected_checkpoint": (
            None
            if selected_seed is None
            else str(root / "training" / f"seed-{selected_seed}" / "final.pt")
        ),
        "decision": "advance_to_player10_with_mastered_replay" if passed else "retain_parent",
        "normal_start_promotable": False,
        "diagnostic_warnings": attribution_warnings,
    }
    temporary = root / ".comparison.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(root / "comparison.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EXP-0016 assisted boss trials")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(compare_assisted_boss(arguments.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
