"""Compare the predeclared EXP-0020 legal Death Metal guide trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autodancer.constants import BossType
from autodancer.curriculum import EpisodeResetSpec
from autodancer.experiments.provenance import sha256_file
from autodancer.training.boss_identity import validate_identity_calibration_report

MODES = ("deterministic", "stochastic-83001", "stochastic-83002")
TRIALS = ("seed-82001", "seed-82002", "seed-82003")
MODE_CONTRACTS = {
    "deterministic": ("deterministic", 0),
    "stochastic-83001": ("stochastic", 83001),
    "stochastic-83002": ("stochastic", 83002),
}
SOURCE_SHA256 = "bdc7d2e2d381cf7ab873d20ff10eafd6e1d15294988c9450d95f253cd3c3dda5"
RESET_SPEC = EpisodeResetSpec("boss-identity", 4, 5, "player20")


def _validated_seed_selection(
    root: Path,
    *,
    bank: str,
    candidates: tuple[int, ...],
    count: int,
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
        raise ValueError(f"EXP-0020 {bank} calibration lacks {count} Death Metal seeds")
    selection_path = (
        root / "training" / "seed-selection.json"
        if bank == "training"
        else root / "evaluation" / "heldout-selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    selected = tuple(int(seed) for seed in selection.get("seeds", ()))
    if selected != expected:
        raise ValueError(f"EXP-0020 {bank} selection does not follow identity-only rule")
    if int(selection.get("boss_type", -1)) != int(BossType.DEATH_METAL):
        raise ValueError(f"EXP-0020 {bank} selection boss mismatch")
    if selection.get("disclosure") != "boss identity only":
        raise ValueError(f"EXP-0020 {bank} selection disclosure mismatch")
    if selection.get("source_report_sha256") != sha256_file(calibration_path):
        raise ValueError(f"EXP-0020 {bank} selection calibration hash mismatch")
    return selected


def _load_report(
    path: Path,
    expected_seeds: tuple[int, ...],
    mode: str,
    *,
    expected_training_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if report.get("controller_valid") is not True:
        raise ValueError(f"controller-invalid evaluation: {path}")
    if int(report.get("worker_restarts", -1)) != 0 or report.get("infrastructure_events"):
        raise ValueError(f"infrastructure-contaminated evaluation: {path}")
    if tuple(int(seed) for seed in report.get("seeds", ())) != expected_seeds:
        raise ValueError(f"held-out seed mismatch: {path}")
    if report.get("curriculum_profile") != "player20":
        raise ValueError(f"curriculum profile mismatch: {path}")
    if int(report.get("curriculum_start_level", 0)) != 4:
        raise ValueError(f"curriculum start mismatch: {path}")
    if int(report.get("curriculum_target_level", 0)) != 5:
        raise ValueError(f"curriculum target mismatch: {path}")
    expected_policy_mode, expected_policy_seed = MODE_CONTRACTS[mode]
    if report.get("policy_mode") != expected_policy_mode:
        raise ValueError(f"policy mode mismatch: {path}")
    if int(report.get("policy_seed", -1)) != expected_policy_seed:
        raise ValueError(f"policy seed mismatch: {path}")
    if report.get("character") != "Bard":
        raise ValueError(f"character mismatch: {path}")
    if report.get("action_contract") != "map-navigation-prior-v1":
        raise ValueError(f"action contract mismatch: {path}")
    if int(report.get("num_instances", 0)) != 8:
        raise ValueError(f"worker capacity mismatch: {path}")
    if int(report.get("max_steps_per_episode", 0)) != 500:
        raise ValueError(f"episode horizon mismatch: {path}")
    if expected_training_seeds is not None:
        if report.get("checkpoint_training_seed_schedule") != "uniform-pool-v1":
            raise ValueError(f"training seed schedule mismatch: {path}")
        if tuple(int(seed) for seed in report.get("checkpoint_training_seed_pool") or ()) != (
            expected_training_seeds
        ):
            raise ValueError(f"training seed pool mismatch: {path}")
        if report.get("checkpoint_curriculum") != {
            "start_level": 4,
            "target_level": 5,
            "profile": "player20",
            "reset_semantics": "normal-reset-sequential-goto-reward-reset-v1",
        }:
            raise ValueError(f"checkpoint curriculum mismatch: {path}")
        if int(report.get("checkpoint_freeze_base_updates", -1)) != 10:
            raise ValueError(f"base-freeze schedule mismatch: {path}")
        initialization = report.get("checkpoint_initialization") or {}
        if initialization.get("sha256") != SOURCE_SHA256:
            raise ValueError(f"initializer checkpoint mismatch: {path}")
        if (
            initialization.get("architecture_upgrade")
            != "v2_to_v8_actor_parity_fresh_critic"
        ):
            raise ValueError(f"initializer architecture transfer mismatch: {path}")
        if int(report.get("checkpoint_global_step", 0)) != 122880:
            raise ValueError(f"training transition budget mismatch: {path}")
    episodes = report.get("trained", {}).get("results", [])
    episode_seeds = tuple(int(episode.get("seed", -1)) for episode in episodes)
    if episode_seeds != expected_seeds:
        raise ValueError(f"episode seed coverage/order mismatch: {path}")
    for episode in episodes:
        seed = int(episode["seed"])
        if int(episode.get("boss_type", 0)) != 2:
            raise ValueError(f"seed {seed} is not Death Metal: {path}")
        actor_types = {int(value) for value in episode.get("boss_actor_types", [])}
        phase_depth = int(episode.get("boss_phase_depth", 0))
        if phase_depth != min(len(actor_types), 4):
            raise ValueError(f"seed {seed} has inconsistent boss phase depth: {path}")
        phase4 = bool(episode.get("death_metal_phase4_reached", False))
        inferred_phase4 = (
            phase_depth >= 4
            and int(episode.get("boss_damage", 0)) >= 7
            and len(actor_types) >= 4
        )
        if phase4 != inferred_phase4:
            raise ValueError(f"seed {seed} has inconsistent phase-4 evidence: {path}")
        if int(episode.get("turns", 0)) > 500:
            raise ValueError(f"seed {seed} exceeded the episode horizon: {path}")
        if episode.get("status") == "curriculum_complete" and int(
            episode.get("furthest_zone", 0)
        ) < 2:
            raise ValueError(f"seed {seed} has a false completion status: {path}")
    return report


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [episode for report in reports for episode in report["trained"]["results"]]
    phase4 = [episode for episode in episodes if episode.get("death_metal_phase4_reached")]
    completions = [
        episode for episode in episodes if episode.get("status") == "curriculum_complete"
    ]
    return {
        "episodes": len(episodes),
        "phase4_count": len(phase4),
        "phase4_rate": len(phase4) / max(len(episodes), 1),
        "distinct_phase4_seeds": sorted({int(episode["seed"]) for episode in phase4}),
        "completion_count": len(completions),
        "completion_rate": len(completions) / max(len(episodes), 1),
        "distinct_completion_seeds": sorted(
            {int(episode["seed"]) for episode in completions}
        ),
        "mean_phase_depth": sum(int(episode.get("boss_phase_depth", 0)) for episode in episodes)
        / max(len(episodes), 1),
        "boss_damage": sum(int(episode.get("boss_damage", 0)) for episode in episodes),
        "death_rate": sum(episode.get("status") == "dead" for episode in episodes)
        / max(len(episodes), 1),
        "mean_turns": sum(int(episode.get("turns", 0)) for episode in episodes)
        / max(len(episodes), 1),
    }


def compare(root: Path) -> dict[str, Any]:
    training_seeds = _validated_seed_selection(
        root,
        bank="training",
        candidates=tuple(range(80001, 80257)),
        count=48,
    )
    seeds = _validated_seed_selection(
        root,
        bank="evaluation",
        candidates=tuple(range(81001, 81257)),
        count=24,
    )
    if set(training_seeds) & set(seeds):
        raise ValueError("EXP-0020 training and held-out seed banks overlap")

    parent_reports = [
        _load_report(root / "evaluation" / "parent" / mode / "report.json", seeds, mode)
        for mode in MODES
    ]
    parent = _aggregate(parent_reports)
    trials: dict[str, dict[str, Any]] = {}
    for trial in TRIALS:
        reports = [
            _load_report(
                root / "evaluation" / trial / mode / "report.json",
                seeds,
                mode,
                expected_training_seeds=training_seeds,
            )
            for mode in MODES
        ]
        trials[trial] = _aggregate(reports)

    all_trial_reports = [
        _load_report(
            root / "evaluation" / trial / mode / "report.json",
            seeds,
            mode,
            expected_training_seeds=training_seeds,
        )
        for trial in TRIALS
        for mode in MODES
    ]
    aggregate = _aggregate(all_trial_reports)
    reproducible_trials = sum(summary["phase4_rate"] >= 0.20 for summary in trials.values())
    boss_damage_improved = aggregate["boss_damage"] / 3 > parent["boss_damage"]
    passed = bool(
        aggregate["phase4_rate"] >= 0.25
        and reproducible_trials >= 2
        and len(aggregate["distinct_phase4_seeds"]) >= 12
        and boss_damage_improved
    )
    ranked = sorted(
        trials,
        key=lambda trial: (
            len(trials[trial]["distinct_completion_seeds"]),
            trials[trial]["phase4_rate"],
            trials[trial]["mean_phase_depth"],
            trials[trial]["boss_damage"],
            -trials[trial]["death_rate"],
            -trials[trial]["mean_turns"],
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "experiment_id": "EXP-0020",
        "controller_valid": True,
        "heldout_seeds": list(seeds),
        "parent": parent,
        "trials": trials,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "phase4_rate_at_least_25_percent": aggregate["phase4_rate"] >= 0.25,
            "at_least_two_reproducible_trials": reproducible_trials >= 2,
            "at_least_12_distinct_phase4_seeds": len(aggregate["distinct_phase4_seeds"])
            >= 12,
            "boss_damage_improved_over_matched_parent": boss_damage_improved,
        },
        "selected_trial": ranked[0] if passed else None,
        "decision": "accept_legal_guide" if passed else "retain_parent_and_diagnose",
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
