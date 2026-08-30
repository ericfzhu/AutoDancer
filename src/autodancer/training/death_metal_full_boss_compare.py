"""Compare the predeclared EXP-0024 full-boss potential trials."""

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

MODES = ("deterministic", "stochastic-95001", "stochastic-95002")
TRIALS = ("seed-94001", "seed-94002", "seed-94003")
MODE_CONTRACTS = {
    "deterministic": ("deterministic", 0),
    "stochastic-95001": ("stochastic", 95001),
    "stochastic-95002": ("stochastic", 95002),
}
SOURCE_SHA256 = "10c1be7bd9e76e4fe3ec7265cc6f712170a3fa1c07b851014c40d2ab111e3b89"
RESET_SPEC = EpisodeResetSpec("boss-identity", 4, 5, "player20")
REWARD_SPEC = load_reward_config(
    Path(__file__).resolve().parents[3] / "configs" / "reward-death-metal-potential-v5.json"
).specification()


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
        raise ValueError(f"EXP-0024 {bank} calibration lacks {count} Death Metal seeds")
    selection_path = (
        root / "training" / "seed-selection.json"
        if bank == "training"
        else root / "evaluation" / "heldout-selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    selected = tuple(int(seed) for seed in selection.get("seeds", ()))
    if selected != expected:
        raise ValueError(f"EXP-0024 {bank} selection violates the identity-only rule")
    if selection.get("disclosure") != "boss identity only":
        raise ValueError(f"EXP-0024 {bank} selection disclosure mismatch")
    if selection.get("source_report_sha256") != sha256_file(calibration_path):
        raise ValueError(f"EXP-0024 {bank} calibration hash mismatch")
    return selected


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
    expected_fields = {
        "character": "Bard",
        "action_contract": "map-navigation-prior-v1",
        "checkpoint_action_contract": "map-navigation-prior-v1",
        "curriculum_profile": "player20",
        "policy_mode": expected_mode,
    }
    for field, expected in expected_fields.items():
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
        raise ValueError(f"episode horizon mismatch: {path}")
    if bool(report.get("source_reference", False)) is not source_reference:
        raise ValueError(f"source-reference identity mismatch: {path}")
    if report.get("evaluation_reward") != REWARD_SPEC:
        raise ValueError(f"evaluation reward mismatch: {path}")
    if report.get("natural_prefix") is not None:
        raise ValueError(f"full-boss evaluation unexpectedly used a guide: {path}")
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
        if report.get("checkpoint_natural_prefix") is not None:
            raise ValueError(f"checkpoint was trained behind an undeclared guide: {path}")
        initialization = report.get("checkpoint_initialization") or {}
        if initialization.get("sha256") != SOURCE_SHA256:
            raise ValueError(f"initializer hash mismatch: {path}")
    episodes = report.get("trained", {}).get("results", [])
    if tuple(int(episode.get("seed", -1)) for episode in episodes) != expected_seeds:
        raise ValueError(f"episode seed coverage/order mismatch: {path}")
    for episode in episodes:
        seed = int(episode["seed"])
        if int(episode.get("boss_type", 0)) != int(BossType.DEATH_METAL):
            raise ValueError(f"seed {seed} is not Death Metal: {path}")
        actor_types = {int(value) for value in episode.get("boss_actor_types", ())}
        if int(episode.get("boss_phase_depth", 0)) != min(len(actor_types), 4):
            raise ValueError(f"seed {seed} has inconsistent phase depth: {path}")
        initial_health = episode.get("initial_boss_health")
        minimum_health = episode.get("minimum_boss_health")
        if initial_health is not None and int(initial_health) != 9:
            raise ValueError(f"seed {seed} did not start from full boss health: {path}")
        if initial_health is not None and minimum_health is not None:
            if not 0 <= int(minimum_health) <= int(initial_health):
                raise ValueError(f"seed {seed} has invalid boss-health evidence: {path}")
        completed = episode.get("status") == "curriculum_complete"
        if completed and int(episode.get("furthest_zone", 0)) < 2:
            raise ValueError(f"seed {seed} has a false Zone 2 completion: {path}")
        if int(episode.get("turns", 0)) > 500:
            raise ValueError(f"seed {seed} exceeded the episode horizon: {path}")
        components = episode.get("reward_components")
        if not isinstance(components, dict):
            raise ValueError(f"seed {seed} lacks reward-component evidence: {path}")
        direct_combat = ("boss_damage", "boss_kill", "enemy_damage", "enemy_kill")
        if any(name in components for name in direct_combat):
            raise ValueError(f"seed {seed} received direct renewable combat shaping: {path}")
    return report


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [episode for report in reports for episode in report["trained"]["results"]]
    completed = [episode for episode in episodes if episode.get("status") == "curriculum_complete"]
    phase_depths = [int(episode.get("boss_phase_depth", 0)) for episode in episodes]
    health_losses = [
        max(
            int(episode.get("initial_boss_health") or 0)
            - int(episode.get("minimum_boss_health") or 0),
            0,
        )
        for episode in episodes
        if episode.get("initial_boss_health") is not None
        and episode.get("minimum_boss_health") is not None
    ]
    return {
        "episodes": len(episodes),
        "completion_count": len(completed),
        "completion_rate": len(completed) / max(len(episodes), 1),
        "distinct_completion_seeds": sorted({int(episode["seed"]) for episode in completed}),
        "mean_phase_depth": sum(phase_depths) / max(len(phase_depths), 1),
        "mean_observed_boss_health_loss": sum(health_losses) / max(len(health_losses), 1),
        "death_rate": sum(episode.get("status") == "dead" for episode in episodes)
        / max(len(episodes), 1),
        "timeout_rate": sum(episode.get("status") == "time_limit" for episode in episodes)
        / max(len(episodes), 1),
        "mean_turns": sum(int(episode.get("turns", 0)) for episode in episodes)
        / max(len(episodes), 1),
    }


def compare(root: Path) -> dict[str, Any]:
    training_seeds = _validated_selection(
        root, bank="training", candidates=tuple(range(92001, 92257)), count=48
    )
    heldout_seeds = _validated_selection(
        root, bank="evaluation", candidates=tuple(range(93001, 93257)), count=24
    )
    if set(training_seeds) & set(heldout_seeds):
        raise ValueError("EXP-0024 training and held-out seed banks overlap")
    parent_reports = [
        _load_report(
            root / "evaluation" / "parent" / mode / "report.json",
            heldout_seeds,
            mode,
            source_reference=True,
        )
        for mode in MODES
    ]
    parent = _aggregate(parent_reports)
    parent["modes"] = {
        mode: _aggregate([report]) for mode, report in zip(MODES, parent_reports, strict=True)
    }
    reports_by_trial: dict[str, list[dict[str, Any]]] = {}
    trials: dict[str, dict[str, Any]] = {}
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
    aggregate = _aggregate([report for reports in reports_by_trial.values() for report in reports])
    reproducible_trials = sum(summary["completion_rate"] >= 0.05 for summary in trials.values())
    every_trial_completes = all(summary["completion_count"] > 0 for summary in trials.values())
    passed = bool(
        aggregate["completion_rate"] >= 0.10
        and reproducible_trials >= 2
        and len(aggregate["distinct_completion_seeds"]) >= 6
        and every_trial_completes
        and aggregate["mean_phase_depth"] > parent["mean_phase_depth"]
        and aggregate["completion_rate"] > parent["completion_rate"]
    )
    ranked = sorted(
        trials,
        key=lambda trial: (
            len(trials[trial]["distinct_completion_seeds"]),
            trials[trial]["completion_rate"],
            trials[trial]["mean_phase_depth"],
            -trials[trial]["death_rate"],
            -trials[trial]["mean_turns"],
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "experiment_id": "EXP-0024",
        "controller_valid": True,
        "heldout_seeds": list(heldout_seeds),
        "parent": parent,
        "trials": trials,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "completion_rate_at_least_10_percent": aggregate["completion_rate"] >= 0.10,
            "at_least_two_reproducible_trials": reproducible_trials >= 2,
            "at_least_six_distinct_completion_seeds": len(
                aggregate["distinct_completion_seeds"]
            )
            >= 6,
            "every_trial_completed_at_least_one_seed": every_trial_completes,
            "mean_phase_depth_improved_over_source": aggregate["mean_phase_depth"]
            > parent["mean_phase_depth"],
            "completion_improved_over_source": aggregate["completion_rate"]
            > parent["completion_rate"],
        },
        "selected_trial": ranked[0] if passed else None,
        "decision": "accept_full_boss_potential" if passed else "reject_full_boss_potential",
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
