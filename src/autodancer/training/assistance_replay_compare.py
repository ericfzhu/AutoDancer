"""Shared comparison logic for assistance-reduction replay experiments."""

from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
    return {
        "final_step": max(int(record.get("global_step", 0)) for record in records),
        "updates": max(int(record.get("updates", 0)) for record in records),
        "finite_losses": finite_losses,
        "positive_gradients": any(
            float(record.get("gradient_norm_preclip", 0.0)) > 0
            for record in records
        ),
        "worker_restarts": max(int(record.get("worker_restarts", 0)) for record in records),
        "collector_recoveries": max(
            int(record.get("collector_recoveries_total", 0)) for record in records
        ),
    }


def _candidate_summary(
    root: Path,
    candidate: str,
    seeds: list[int],
    *,
    profiles: tuple[str, ...],
    modes: tuple[tuple[str, str, int], ...],
) -> dict[str, Any]:
    profile_summaries: dict[str, Any] = {}
    candidate_valid = True
    for profile in profiles:
        reports: dict[str, Any] = {}
        sampled_successes = 0
        sampled_episodes = 0
        sampled_deaths = 0
        sampled_seeds: set[int] = set()
        sampled_turns: list[int] = []
        deterministic_rate = 0.0
        for name, expected_mode, policy_seed in modes:
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
        profile_summaries[profile] = {
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
    return {"controller_valid": candidate_valid, "profiles": profile_summaries}


def compare_assistance_replay(
    root: Path,
    *,
    experiment_id: str,
    candidates: tuple[str, ...],
    profiles: tuple[str, ...],
    modes: tuple[tuple[str, str, int], ...],
    primary_profile: str,
    replay_profile: str,
    primary_label: str,
    expected_training_steps: int,
    minimum_mean_completion: float,
    minimum_individual_completion: float,
    maximum_individual_death: float,
    minimum_replay_retention: float,
    pass_decision: str,
    fail_decision: str,
) -> dict[str, Any]:
    """Apply a reproducibility, health, and retained-skill gate."""
    trained = candidates[1:]
    selection = json.loads(
        (root / "evaluation" / "heldout-selection.json").read_text(
            encoding="utf-8-sig"
        )
    )
    seeds = [int(seed) for seed in selection["seeds"]]
    if len(seeds) != 24 or len(set(seeds)) != 24:
        raise ValueError(f"{experiment_id} final evaluation requires 24 distinct seeds")
    summaries = {
        candidate: _candidate_summary(
            root, candidate, seeds, profiles=profiles, modes=modes
        )
        for candidate in candidates
    }
    for candidate in trained:
        summaries[candidate]["training"] = _training_health(
            root / "training" / candidate / "metrics.jsonl"
        )

    parent_primary = summaries["parent"]["profiles"][primary_profile][
        "sampled_completion_rate"
    ]
    parent_replay = summaries["parent"]["profiles"][replay_profile][
        "sampled_completion_rate"
    ]
    improved: list[str] = []
    eligible: list[str] = []
    for candidate in trained:
        value = summaries[candidate]
        health = value["training"]
        primary = value["profiles"][primary_profile]
        replay = value["profiles"][replay_profile]
        healthy = (
            health["final_step"] == expected_training_steps
            and health["finite_losses"]
            and health["positive_gradients"]
            and health["worker_restarts"] == 0
            and health["collector_recoveries"] == 0
            and value["controller_valid"]
        )
        improves = primary["sampled_completion_rate"] > parent_primary
        retained = replay["sampled_completion_rate"] >= (
            minimum_replay_retention * parent_replay
        )
        individually_eligible = (
            healthy
            and primary["sampled_completion_rate"] >= minimum_individual_completion
            and primary["sampled_death_rate"] <= maximum_individual_death
            and retained
        )
        value["healthy"] = healthy
        value[f"improves_parent_{primary_label}"] = improves
        value["replay_skill_retained"] = retained
        value["individually_eligible"] = individually_eligible
        if improves:
            improved.append(candidate)
        if individually_eligible:
            eligible.append(candidate)

    mean_primary = statistics.mean(
        summaries[candidate]["profiles"][primary_profile]["sampled_completion_rate"]
        for candidate in trained
    )
    passed = (
        mean_primary >= minimum_mean_completion
        and len(improved) >= 2
        and bool(eligible)
    )

    def rank(candidate: str) -> tuple[float, ...]:
        value = summaries[candidate]
        primary = value["profiles"][primary_profile]
        replay = value["profiles"][replay_profile]
        median_turns = primary["median_sampled_success_turns"]
        return (
            float(len(primary["distinct_successful_seeds"])),
            float(primary["sampled_completion_rate"]),
            -float(primary["sampled_death_rate"]),
            float(replay["sampled_completion_rate"]),
            float(primary["deterministic_completion_rate"]),
            -float(median_turns if median_turns is not None else math.inf),
        )

    selected = max(eligible, key=rank) if passed else None
    result = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "seeds": seeds,
        "candidates": summaries,
        f"parent_{primary_label}_sampled_completion": parent_primary,
        "parent_replay_sampled_completion": parent_replay,
        f"mean_trained_{primary_label}_sampled_completion": mean_primary,
        "thresholds": {
            "minimum_mean_completion": minimum_mean_completion,
            "minimum_individual_completion": minimum_individual_completion,
            "maximum_individual_death": maximum_individual_death,
            "minimum_replay_retention": minimum_replay_retention,
        },
        "improved_trials": improved,
        "eligible_trials": eligible,
        "passed": passed,
        "selected_trial": selected,
        "selected_checkpoint": (
            None if selected is None else str(root / "training" / selected / "final.pt")
        ),
        "decision": pass_decision if passed else fail_decision,
        "normal_start_promotable": False,
    }
    temporary = root / ".comparison.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(root / "comparison.json")
    return result
