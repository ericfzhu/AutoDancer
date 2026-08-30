"""Compare the predeclared EXP-0023 legal phase-3 successor trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autodancer.constants import BossType
from autodancer.curriculum import EpisodeResetSpec
from autodancer.experiments.provenance import sha256_file
from autodancer.rewards import load_reward_config
from autodancer.training.boss_identity import validate_identity_calibration_report

MODES = ("deterministic", "stochastic-91021", "stochastic-91022")
TRIALS = ("seed-90001", "seed-90002", "seed-90003")
MODE_CONTRACTS = {
    "deterministic": ("deterministic", 0),
    "stochastic-91021": ("stochastic", 91021),
    "stochastic-91022": ("stochastic", 91022),
}
SOURCE_SHA256 = "10c1be7bd9e76e4fe3ec7265cc6f712170a3fa1c07b851014c40d2ab111e3b89"
RESET_SPEC = EpisodeResetSpec("boss-identity", 4, 5, "player20")
REWARD_SPEC = load_reward_config(
    Path(__file__).resolve().parents[3] / "configs" / "reward-death-metal-potential-v5.json"
).specification()


def _prefix_contract() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "death-metal-natural-prefix-v2",
        "boss_type": int(BossType.DEATH_METAL),
        "target_phase": 3,
        "max_guide_turns": 500,
        "max_attempts": 8,
        "max_failed_seeds_per_fragment": 16,
        "deterministic_guide": False,
        "guide_policy_seed": 87001,
        "recurrent_state_mode": "warm",
        "target_health": 4,
        "state_semantics": "ordinary-engine-transitions-only",
        "guide_transitions_in_ppo": False,
    }


def _validated_selection(
    root: Path, *, bank: str, candidates: tuple[int, ...], count: int
) -> tuple[int, ...]:
    calibration_path = root / "calibration" / f"{bank}-candidates.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    results = validate_identity_calibration_report(
        calibration,
        expected_seeds=candidates,
        reset_spec=RESET_SPEC,
        num_instances=8,
    )
    expected = tuple(
        sorted(
            int(result["seed"])
            for result in results
            if int(result["boss_type"]) == int(BossType.DEATH_METAL)
        )[:count]
    )
    if len(expected) != count:
        raise ValueError(f"EXP-0023 {bank} calibration lacks {count} Death Metal seeds")
    selection_path = (
        root / "training" / "seed-selection.json"
        if bank == "training"
        else root / "evaluation" / "heldout-selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    selected = tuple(int(seed) for seed in selection.get("seeds", ()))
    if selected != expected:
        raise ValueError(f"EXP-0023 {bank} selection violates the identity-only rule")
    if selection.get("disclosure") != "boss identity only":
        raise ValueError(f"EXP-0023 {bank} selection disclosure mismatch")
    if selection.get("source_report_sha256") != sha256_file(calibration_path):
        raise ValueError(f"EXP-0023 {bank} calibration hash mismatch")
    return selected


def _validate_prefix(value: Any, *, path: Path) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"natural-prefix contract missing: {path}")
    expected = _prefix_contract()
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(f"natural-prefix {field} mismatch: {path}")
    if value.get("guide_checkpoint_sha256") != SOURCE_SHA256:
        raise ValueError(f"natural-prefix guide hash mismatch: {path}")


def _load_report(
    path: Path,
    expected_seeds: tuple[int, ...],
    mode: str,
    *,
    expected_training_seeds: tuple[int, ...] | None = None,
    source_reference: bool = False,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if report.get("controller_valid") is not True:
        raise ValueError(f"controller-invalid evaluation: {path}")
    if int(report.get("worker_restarts", -1)) != 0 or report.get("infrastructure_events"):
        raise ValueError(f"infrastructure-contaminated evaluation: {path}")
    if tuple(int(seed) for seed in report.get("seeds", ())) != expected_seeds:
        raise ValueError(f"held-out seed mismatch: {path}")
    expected_mode, expected_policy_seed = MODE_CONTRACTS[mode]
    expected_top_level = {
        "character": "Bard",
        "action_contract": "map-navigation-prior-v1",
        "checkpoint_action_contract": "map-navigation-prior-v1",
        "curriculum_profile": "player20",
        "policy_mode": expected_mode,
    }
    for field, expected in expected_top_level.items():
        if report.get(field) != expected:
            raise ValueError(f"{field} mismatch: {path}")
    if int(report.get("policy_seed", -1)) != expected_policy_seed:
        raise ValueError(f"policy seed mismatch: {path}")
    if int(report.get("curriculum_start_level", 0)) != 4:
        raise ValueError(f"curriculum start mismatch: {path}")
    if int(report.get("curriculum_target_level", 0)) != 5:
        raise ValueError(f"curriculum target mismatch: {path}")
    if int(report.get("num_instances", 0)) != 8:
        raise ValueError(f"worker capacity mismatch: {path}")
    if int(report.get("max_steps_per_episode", 0)) != 500:
        raise ValueError(f"learner horizon mismatch: {path}")
    if bool(report.get("source_reference", False)) is not source_reference:
        raise ValueError(f"source-reference identity mismatch: {path}")
    if report.get("evaluation_reward") != REWARD_SPEC:
        raise ValueError(f"evaluation reward mismatch: {path}")
    _validate_prefix(report.get("natural_prefix"), path=path)
    if expected_training_seeds is not None:
        if report.get("reward") != REWARD_SPEC:
            raise ValueError(f"checkpoint reward mismatch: {path}")
        if tuple(int(seed) for seed in report.get("checkpoint_training_seed_pool") or ()) != (
            expected_training_seeds
        ):
            raise ValueError(f"training seed pool mismatch: {path}")
        if int(report.get("checkpoint_global_step", 0)) != 122880:
            raise ValueError(f"training budget mismatch: {path}")
        if int(report.get("checkpoint_freeze_base_updates", -1)) != 10:
            raise ValueError(f"base-freeze schedule mismatch: {path}")
        _validate_prefix(report.get("checkpoint_natural_prefix"), path=path)
        initialization = report.get("checkpoint_initialization") or {}
        if initialization.get("sha256") != SOURCE_SHA256:
            raise ValueError(f"initializer hash mismatch: {path}")
    episodes = report.get("trained", {}).get("results", [])
    if tuple(int(episode.get("seed", -1)) for episode in episodes) != expected_seeds:
        raise ValueError(f"episode seed coverage/order mismatch: {path}")
    for episode in episodes:
        seed = int(episode["seed"])
        prefix = episode.get("natural_prefix") or {}
        acquired = bool(prefix.get("acquired", False))
        boundary = prefix.get("boundary") or {}
        if acquired:
            if not bool(boundary.get("reached")):
                raise ValueError(f"seed {seed} has an unproven handoff: {path}")
            if int(boundary.get("target_phase", 0)) != 3:
                raise ValueError(f"seed {seed} has wrong handoff phase: {path}")
            if int(boundary.get("minimum_health", 99)) > 4:
                raise ValueError(f"seed {seed} has wrong handoff health: {path}")
            if len({int(value) for value in boundary.get("observed_actor_types", ())}) < 3:
                raise ValueError(f"seed {seed} lacks phase-entity evidence: {path}")
        elif episode.get("status") != "prefix_failed":
            raise ValueError(f"seed {seed} lost a failed acquisition from its outcome: {path}")
        completed = episode.get("status") == "curriculum_complete"
        if completed and (not acquired or int(episode.get("furthest_zone", 0)) < 2):
            raise ValueError(f"seed {seed} has a false Zone 2 completion: {path}")
        if int(episode.get("turns", 0)) > 500:
            raise ValueError(f"seed {seed} exceeded the learner horizon: {path}")
        components = episode.get("reward_components")
        if not isinstance(components, dict):
            raise ValueError(f"seed {seed} lacks reward-component evidence: {path}")
        direct_combat_components = ("boss_damage", "boss_kill", "enemy_damage", "enemy_kill")
        if any(name in components for name in direct_combat_components):
            raise ValueError(f"seed {seed} received direct renewable combat shaping: {path}")
    return report


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [episode for report in reports for episode in report["trained"]["results"]]
    acquired = [
        episode for episode in episodes if episode.get("natural_prefix", {}).get("acquired")
    ]
    completed = [episode for episode in episodes if episode.get("status") == "curriculum_complete"]
    deaths = [episode for episode in episodes if episode.get("status") == "dead"]
    action_counts = [0] * 11
    for episode in acquired:
        for index, count in enumerate(episode.get("action_counts", ())):
            action_counts[index] += int(count)
    actions = max(sum(action_counts), 1)
    return {
        "episodes": len(episodes),
        "acquired_count": len(acquired),
        "acquisition_rate": len(acquired) / max(len(episodes), 1),
        "completion_count": len(completed),
        "unconditional_completion_rate": len(completed) / max(len(episodes), 1),
        "conditional_completion_rate": len(completed) / max(len(acquired), 1),
        "distinct_completion_seeds": sorted({int(episode["seed"]) for episode in completed}),
        "death_rate_after_acquisition": len(deaths) / max(len(acquired), 1),
        "mean_learner_turns_after_acquisition": sum(
            int(episode.get("turns", 0)) for episode in acquired
        )
        / max(len(acquired), 1),
        "mean_guide_turns": sum(
            int(episode.get("natural_prefix", {}).get("guide_turns", 0)) for episode in episodes
        )
        / max(len(episodes), 1),
        "wait_rate": action_counts[4] / actions,
        "action_counts": action_counts,
    }


def compare(root: Path) -> dict[str, Any]:
    training_seeds = _validated_selection(
        root, bank="training", candidates=tuple(range(88001, 88257)), count=48
    )
    heldout_seeds = _validated_selection(
        root, bank="evaluation", candidates=tuple(range(89001, 89257)), count=24
    )
    if set(training_seeds) & set(heldout_seeds):
        raise ValueError("EXP-0023 training and held-out seed banks overlap")
    parent_reports = [
        _load_report(
            root / "evaluation" / "parent" / mode / "report.json",
            heldout_seeds,
            mode,
            source_reference=True,
        )
        for mode in MODES
    ]
    trials: dict[str, dict[str, Any]] = {}
    reports_by_trial: dict[str, list[dict[str, Any]]] = {}
    for trial in TRIALS:
        reports = [
            _load_report(
                root / "evaluation" / trial / mode / "report.json",
                heldout_seeds,
                mode,
                expected_training_seeds=training_seeds,
            )
            for mode in MODES
        ]
        reports_by_trial[trial] = reports
        trials[trial] = _aggregate(reports)
        trials[trial]["modes"] = {
            mode: _aggregate([report]) for mode, report in zip(MODES, reports, strict=True)
        }
    parent = _aggregate(parent_reports)
    parent["modes"] = {
        mode: _aggregate([report]) for mode, report in zip(MODES, parent_reports, strict=True)
    }
    aggregate = _aggregate([report for reports in reports_by_trial.values() for report in reports])
    reproducible_trials = sum(
        summary["unconditional_completion_rate"] >= 0.10 for summary in trials.values()
    )
    every_trial_completes = all(summary["completion_count"] > 0 for summary in trials.values())
    passed = bool(
        aggregate["acquisition_rate"] >= 0.50
        and aggregate["unconditional_completion_rate"] >= 0.15
        and aggregate["conditional_completion_rate"] >= 0.25
        and reproducible_trials >= 2
        and len(aggregate["distinct_completion_seeds"]) >= 8
        and every_trial_completes
    )
    ranked = sorted(
        trials,
        key=lambda trial: (
            len(trials[trial]["distinct_completion_seeds"]),
            trials[trial]["unconditional_completion_rate"],
            trials[trial]["conditional_completion_rate"],
            trials[trial]["acquisition_rate"],
            -trials[trial]["death_rate_after_acquisition"],
            -trials[trial]["mean_learner_turns_after_acquisition"],
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "experiment_id": "EXP-0023",
        "controller_valid": True,
        "heldout_seeds": list(heldout_seeds),
        "parent": parent,
        "trials": trials,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "acquisition_rate_at_least_50_percent": aggregate["acquisition_rate"] >= 0.50,
            "unconditional_completion_at_least_15_percent": aggregate[
                "unconditional_completion_rate"
            ]
            >= 0.15,
            "conditional_completion_at_least_25_percent": aggregate[
                "conditional_completion_rate"
            ]
            >= 0.25,
            "at_least_two_reproducible_trials": reproducible_trials >= 2,
            "at_least_eight_distinct_completion_seeds": len(
                aggregate["distinct_completion_seeds"]
            )
            >= 8,
            "every_trial_completed_at_least_one_seed": every_trial_completes,
        },
        "selected_trial": ranked[0] if passed else None,
        "decision": "accept_phase3_successor" if passed else "reject_phase3_successor",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare(arguments.root)
    output = arguments.root / "comparison.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result["gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
