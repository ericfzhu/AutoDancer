"""Compare the conditional EXP-0018 mixed player6 replay trials."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CANDIDATES = ("parent", "seed-73001", "seed-73002", "seed-73003")
TRAINED = CANDIDATES[1:]
PROFILES = ("boss1hp-player6", "boss1hp-player10")
MODES = (
    ("deterministic", "deterministic", 0),
    ("stochastic-104001", "stochastic", 104001),
    ("stochastic-104002", "stochastic", 104002),
)


def _success(result: dict[str, Any]) -> bool:
    return (
        str(result.get("status")) == "curriculum_complete"
        and int(result.get("furthest_zone", 0)) >= 2
    )


def _training_health(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"training metrics are empty: {path}")
    finite_losses = all(
        math.isfinite(float(record[name]))
        for record in records
        for name in ("policy_loss", "value_loss", "entropy", "gradient_norm_preclip")
    )
    positive_gradients = any(
        float(record.get("gradient_norm_preclip", 0.0)) > 0 for record in records
    )
    return {
        "final_step": max(int(record.get("global_step", 0)) for record in records),
        "updates": max(int(record.get("updates", 0)) for record in records),
        "finite_losses": finite_losses,
        "positive_gradients": positive_gradients,
        "worker_restarts": max(int(record.get("worker_restarts", 0)) for record in records),
        "collector_recoveries": max(
            int(record.get("collector_recoveries_total", 0)) for record in records
        ),
    }


def _candidate_summary(root: Path, candidate: str, seeds: list[int]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    candidate_valid = True
    for profile in PROFILES:
        reports: dict[str, Any] = {}
        sampled_successes = 0
        sampled_episodes = 0
        sampled_deaths = 0
        sampled_seeds: set[int] = set()
        sampled_turns: list[int] = []
        deterministic_rate = 0.0
        for name, expected_mode, policy_seed in MODES:
            path = root / "evaluation" / candidate / profile / name / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            results = list(report["trained"]["results"])
            result_seeds = [int(result["seed"]) for result in results]
            valid = bool(report.get("controller_valid"))
            valid = valid and not report.get("infrastructure_events")
            valid = valid and int(report.get("worker_restarts", 0)) == 0
            valid = valid and report.get("policy_mode") == expected_mode
            valid = valid and int(report.get("policy_seed", -1)) == policy_seed
            valid = valid and report.get("curriculum_profile") == profile
            valid = valid and result_seeds == seeds
            successes = [result for result in results if _success(result)]
            deaths = sum(str(result.get("status")) == "dead" for result in results)
            successful_seeds = sorted(int(result["seed"]) for result in successes)
            rate = len(successes) / max(len(results), 1)
            if expected_mode == "deterministic":
                deterministic_rate = rate
            else:
                sampled_successes += len(successes)
                sampled_episodes += len(results)
                sampled_deaths += deaths
                sampled_seeds.update(successful_seeds)
                sampled_turns.extend(int(result.get("turns", 0)) for result in successes)
            reports[name] = {
                "valid": valid,
                "episodes": len(results),
                "successes": len(successes),
                "completion_rate": rate,
                "deaths": deaths,
                "successful_seeds": successful_seeds,
            }
            candidate_valid = candidate_valid and valid
        profiles[profile] = {
            "reports": reports,
            "sampled_completion_rate": sampled_successes / max(sampled_episodes, 1),
            "sampled_death_rate": sampled_deaths / max(sampled_episodes, 1),
            "sampled_successes": sampled_successes,
            "sampled_episodes": sampled_episodes,
            "distinct_successful_seeds": sorted(sampled_seeds),
            "deterministic_completion_rate": deterministic_rate,
            "median_sampled_success_turns": (
                statistics.median(sampled_turns) if sampled_turns else None
            ),
        }
    return {"controller_valid": candidate_valid, "profiles": profiles}


def compare_player6_replay(root: Path) -> dict[str, Any]:
    selection = json.loads(
        (root / "evaluation" / "heldout-selection.json").read_text(encoding="utf-8-sig")
    )
    seeds = [int(seed) for seed in selection["seeds"]]
    if len(seeds) != 24 or len(set(seeds)) != 24:
        raise ValueError("EXP-0018 final evaluation requires 24 distinct seeds")
    candidates = {
        candidate: _candidate_summary(root, candidate, seeds) for candidate in CANDIDATES
    }
    for candidate in TRAINED:
        candidates[candidate]["training"] = _training_health(
            root / "training" / candidate / "metrics.jsonl"
        )

    player6 = "boss1hp-player6"
    player10 = "boss1hp-player10"
    parent_player6 = candidates["parent"]["profiles"][player6][
        "sampled_completion_rate"
    ]
    parent_player10 = candidates["parent"]["profiles"][player10][
        "sampled_completion_rate"
    ]
    improved = []
    eligible = []
    for candidate in TRAINED:
        value = candidates[candidate]
        health = value["training"]
        six = value["profiles"][player6]
        ten = value["profiles"][player10]
        healthy = (
            health["final_step"] == 51200
            and health["finite_losses"]
            and health["positive_gradients"]
            and health["worker_restarts"] == 0
            and health["collector_recoveries"] == 0
            and value["controller_valid"]
        )
        improves = six["sampled_completion_rate"] > parent_player6
        retained = ten["sampled_completion_rate"] >= 0.8 * parent_player10
        individually_eligible = (
            healthy
            and six["sampled_completion_rate"] >= 0.5
            and six["sampled_death_rate"] <= 0.5
            and retained
        )
        value["healthy"] = healthy
        value["improves_parent_player6"] = improves
        value["player10_retained"] = retained
        value["individually_eligible"] = individually_eligible
        improved.extend([candidate] if improves else [])
        eligible.extend([candidate] if individually_eligible else [])

    mean_player6 = statistics.mean(
        candidates[candidate]["profiles"][player6]["sampled_completion_rate"]
        for candidate in TRAINED
    )
    passed = mean_player6 >= 0.5 and len(improved) >= 2 and bool(eligible)

    def rank(candidate: str) -> tuple[float, ...]:
        value = candidates[candidate]
        six = value["profiles"][player6]
        ten = value["profiles"][player10]
        median_turns = six["median_sampled_success_turns"]
        return (
            float(len(six["distinct_successful_seeds"])),
            float(six["sampled_completion_rate"]),
            -float(six["sampled_death_rate"]),
            float(ten["sampled_completion_rate"]),
            float(six["deterministic_completion_rate"]),
            -float(median_turns if median_turns is not None else math.inf),
        )

    selected = max(eligible, key=rank) if passed else None
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0018",
        "created_at": datetime.now(UTC).isoformat(),
        "seeds": seeds,
        "candidates": candidates,
        "parent_player6_sampled_completion": parent_player6,
        "parent_player10_sampled_completion": parent_player10,
        "mean_trained_player6_sampled_completion": mean_player6,
        "improved_trials": improved,
        "eligible_trials": eligible,
        "passed": passed,
        "selected_trial": selected,
        "selected_checkpoint": (
            None if selected is None else str(root / "training" / selected / "final.pt")
        ),
        "decision": "advance_to_boss_health" if passed else "retain_parent",
        "normal_start_promotable": False,
    }
    temporary = root / ".comparison.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(root / "comparison.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EXP-0018 replay trials")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(compare_player6_replay(arguments.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
