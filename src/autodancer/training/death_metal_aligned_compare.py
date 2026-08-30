"""Compare the predeclared EXP-0022 outcome-aligned Death Metal guide trials."""

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

MODES = ("deterministic", "stochastic-87001", "stochastic-87002")
TRIALS = ("seed-86001", "seed-86002", "seed-86003")
MODE_CONTRACTS = {
    "deterministic": ("deterministic", 0),
    "stochastic-87001": ("stochastic", 87001),
    "stochastic-87002": ("stochastic", 87002),
}
SOURCE_SHA256 = "bdc7d2e2d381cf7ab873d20ff10eafd6e1d15294988c9450d95f253cd3c3dda5"
RESET_SPEC = EpisodeResetSpec("boss-identity", 4, 5, "player20")
GUIDE_REWARD_SPEC = load_reward_config(
    Path(__file__).resolve().parents[3] / "configs" / "reward-death-metal-guide-v3.json"
).specification()


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
        raise ValueError(f"EXP-0022 {bank} calibration lacks {count} Death Metal seeds")
    selection_path = (
        root / "training" / "seed-selection.json"
        if bank == "training"
        else root / "evaluation" / "heldout-selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    selected = tuple(int(seed) for seed in selection.get("seeds", ()))
    if selected != expected:
        raise ValueError(f"EXP-0022 {bank} selection does not follow identity-only rule")
    if int(selection.get("boss_type", -1)) != int(BossType.DEATH_METAL):
        raise ValueError(f"EXP-0022 {bank} selection boss mismatch")
    if selection.get("disclosure") != "boss identity only":
        raise ValueError(f"EXP-0022 {bank} selection disclosure mismatch")
    if selection.get("source_report_sha256") != sha256_file(calibration_path):
        raise ValueError(f"EXP-0022 {bank} selection calibration hash mismatch")
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
    expected_top_level = {
        "character": "Bard",
        "action_contract": "map-navigation-prior-v1",
        "checkpoint_action_contract": "map-navigation-prior-v1",
        "curriculum_profile": "player20",
    }
    for field, expected in expected_top_level.items():
        if report.get(field) != expected:
            raise ValueError(f"{field} mismatch: {path}")
    if int(report.get("curriculum_start_level", 0)) != 4:
        raise ValueError(f"curriculum start mismatch: {path}")
    if int(report.get("curriculum_target_level", 0)) != 5:
        raise ValueError(f"curriculum target mismatch: {path}")
    if report.get("policy_mode") != expected_mode:
        raise ValueError(f"policy mode mismatch: {path}")
    if int(report.get("policy_seed", -1)) != expected_policy_seed:
        raise ValueError(f"policy seed mismatch: {path}")
    if report.get("evaluation_reward") != GUIDE_REWARD_SPEC:
        raise ValueError(f"guide evaluation reward mismatch: {path}")
    if int(report.get("num_instances", 0)) != 8:
        raise ValueError(f"worker capacity mismatch: {path}")
    if int(report.get("max_steps_per_episode", 0)) != 500:
        raise ValueError(f"episode horizon mismatch: {path}")
    if bool(report.get("source_reference", False)) is not source_reference:
        raise ValueError(f"source-reference identity mismatch: {path}")
    if expected_training_seeds is not None:
        if report.get("reward") != GUIDE_REWARD_SPEC:
            raise ValueError(f"checkpoint guide reward mismatch: {path}")
        if report.get("checkpoint_training_seed_schedule") != "uniform-pool-v1":
            raise ValueError(f"training seed schedule mismatch: {path}")
        if tuple(int(seed) for seed in report.get("checkpoint_training_seed_pool") or ()) != (
            expected_training_seeds
        ):
            raise ValueError(f"training seed pool mismatch: {path}")
        expected_curriculum = {
            "start_level": 4,
            "target_level": 5,
            "profile": "player20",
            "reset_semantics": "normal-reset-sequential-goto-reward-reset-v1",
        }
        if report.get("checkpoint_curriculum") != expected_curriculum:
            raise ValueError(f"checkpoint curriculum mismatch: {path}")
        if int(report.get("checkpoint_freeze_base_updates", -1)) != 10:
            raise ValueError(f"base-freeze schedule mismatch: {path}")
        if report.get("checkpoint_freeze_base_scope") != "inherited-actor-base-only-v1":
            raise ValueError(f"base-freeze scope mismatch: {path}")
        initialization = report.get("checkpoint_initialization") or {}
        if initialization.get("sha256") != SOURCE_SHA256:
            raise ValueError(f"initializer checkpoint mismatch: {path}")
        if initialization.get("architecture_upgrade") != "v2_to_v8_actor_parity_fresh_critic":
            raise ValueError(f"initializer architecture transfer mismatch: {path}")
        if int(report.get("checkpoint_global_step", 0)) != 122880:
            raise ValueError(f"training transition budget mismatch: {path}")
    episodes = report.get("trained", {}).get("results", [])
    if tuple(int(episode.get("seed", -1)) for episode in episodes) != expected_seeds:
        raise ValueError(f"episode seed coverage/order mismatch: {path}")
    for episode in episodes:
        seed = int(episode["seed"])
        if int(episode.get("boss_type", 0)) != int(BossType.DEATH_METAL):
            raise ValueError(f"seed {seed} is not Death Metal: {path}")
        actor_types = {int(value) for value in episode.get("boss_actor_types", [])}
        phase_depth = int(episode.get("boss_phase_depth", 0))
        if phase_depth != min(len(actor_types), 4):
            raise ValueError(f"seed {seed} has inconsistent phase depth: {path}")
        inferred_phase4 = (
            phase_depth >= 4 and int(episode.get("boss_damage", 0)) >= 7 and len(actor_types) >= 4
        )
        if bool(episode.get("death_metal_phase4_reached", False)) != inferred_phase4:
            raise ValueError(f"seed {seed} has inconsistent phase-4 evidence: {path}")
        if int(episode.get("turns", 0)) > 500:
            raise ValueError(f"seed {seed} exceeded the episode horizon: {path}")
        if (
            episode.get("status") == "curriculum_complete"
            and int(episode.get("furthest_zone", 0)) < 2
        ):
            raise ValueError(f"seed {seed} has a false completion status: {path}")
        components = episode.get("reward_components")
        if not isinstance(components, dict):
            raise ValueError(f"seed {seed} lacks reward-component evidence: {path}")
        if "enemy_damage" in components or "enemy_kill" in components:
            raise ValueError(f"seed {seed} received non-boss combat shaping: {path}")
        if abs(float(components.get("player_damage", 0.0))) > 1e-9:
            raise ValueError(f"seed {seed} received player-damage shaping: {path}")
        if abs(float(components.get("death", 0.0))) > 1e-9:
            raise ValueError(f"seed {seed} received death shaping: {path}")
        maximum_boss_credit = (
            int(episode.get("boss_damage", 0)) * 0.2 + int(episode.get("boss_kills", 0)) * 0.25
        )
        observed_boss_credit = float(components.get("boss_damage", 0.0)) + float(
            components.get("boss_kill", 0.0)
        )
        if observed_boss_credit > maximum_boss_credit + 1e-9:
            raise ValueError(f"seed {seed} has unsupported boss credit: {path}")
    return report


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [episode for report in reports for episode in report["trained"]["results"]]
    phase2 = [episode for episode in episodes if int(episode.get("boss_phase_depth", 0)) >= 2]
    phase3 = [episode for episode in episodes if int(episode.get("boss_phase_depth", 0)) >= 3]
    phase4 = [episode for episode in episodes if episode.get("death_metal_phase4_reached")]
    completions = [
        episode for episode in episodes if episode.get("status") == "curriculum_complete"
    ]
    zero_contact_timeouts = [
        episode
        for episode in episodes
        if episode.get("status") == "time_limit" and int(episode.get("boss_damage", 0)) == 0
    ]
    action_counts = [0] * 11
    for episode in episodes:
        for index, count in enumerate(episode.get("action_counts", ())):
            action_counts[index] += int(count)
    actions = max(sum(action_counts), 1)
    return {
        "episodes": len(episodes),
        "phase2_count": len(phase2),
        "phase2_rate": len(phase2) / max(len(episodes), 1),
        "distinct_phase2_seeds": sorted({int(episode["seed"]) for episode in phase2}),
        "phase3_count": len(phase3),
        "phase3_rate": len(phase3) / max(len(episodes), 1),
        "distinct_phase3_seeds": sorted({int(episode["seed"]) for episode in phase3}),
        "phase4_count": len(phase4),
        "phase4_rate": len(phase4) / max(len(episodes), 1),
        "distinct_phase4_seeds": sorted({int(episode["seed"]) for episode in phase4}),
        "completion_count": len(completions),
        "completion_rate": len(completions) / max(len(episodes), 1),
        "distinct_completion_seeds": sorted({int(episode["seed"]) for episode in completions}),
        "mean_phase_depth": sum(int(episode.get("boss_phase_depth", 0)) for episode in episodes)
        / max(len(episodes), 1),
        "boss_damage": sum(int(episode.get("boss_damage", 0)) for episode in episodes),
        "death_rate": sum(episode.get("status") == "dead" for episode in episodes)
        / max(len(episodes), 1),
        "step_limit_rate": sum(episode.get("status") == "time_limit" for episode in episodes)
        / max(len(episodes), 1),
        "zero_contact_timeout_rate": len(zero_contact_timeouts) / max(len(episodes), 1),
        "mean_turns": sum(int(episode.get("turns", 0)) for episode in episodes)
        / max(len(episodes), 1),
        "wait_rate": action_counts[4] / actions,
        "movement_rate": sum(action_counts[:4]) / actions,
        "action_counts": action_counts,
    }


def compare(root: Path) -> dict[str, Any]:
    training_seeds = _validated_seed_selection(
        root, bank="training", candidates=tuple(range(84001, 84257)), count=48
    )
    heldout_seeds = _validated_seed_selection(
        root, bank="evaluation", candidates=tuple(range(85001, 85257)), count=24
    )
    if set(training_seeds) & set(heldout_seeds):
        raise ValueError("EXP-0022 training and held-out seed banks overlap")
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
    reproducible_trials = sum(summary["phase4_rate"] >= 0.20 for summary in trials.values())
    all_trials_contact = all(summary["boss_damage"] > 0 for summary in trials.values())
    boss_damage_improved = aggregate["boss_damage"] / len(TRIALS) > parent["boss_damage"]
    passed = bool(
        aggregate["phase4_rate"] >= 0.25
        and reproducible_trials >= 2
        and len(aggregate["distinct_phase4_seeds"]) >= 12
        and all_trials_contact
        and boss_damage_improved
    )
    ranked = sorted(
        trials,
        key=lambda trial: (
            len(trials[trial]["distinct_completion_seeds"]),
            trials[trial]["phase4_rate"],
            trials[trial]["mean_phase_depth"],
            trials[trial]["boss_damage"],
            -trials[trial]["zero_contact_timeout_rate"],
            -trials[trial]["death_rate"],
            -trials[trial]["mean_turns"],
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "experiment_id": "EXP-0022",
        "controller_valid": True,
        "heldout_seeds": list(heldout_seeds),
        "parent": parent,
        "trials": trials,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "phase4_rate_at_least_25_percent": aggregate["phase4_rate"] >= 0.25,
            "at_least_two_reproducible_trials": reproducible_trials >= 2,
            "at_least_12_distinct_phase4_seeds": len(aggregate["distinct_phase4_seeds"]) >= 12,
            "every_trial_retained_boss_contact": all_trials_contact,
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
