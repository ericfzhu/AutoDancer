"""Compare the predeclared EXP-0020 legal Death Metal guide trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODES = ("deterministic", "stochastic-83001", "stochastic-83002")
TRIALS = ("seed-82001", "seed-82002", "seed-82003")


def _load_report(path: Path, expected_seeds: tuple[int, ...]) -> dict[str, Any]:
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
    selection = json.loads(
        (root / "evaluation" / "heldout-selection.json").read_text(encoding="utf-8-sig")
    )
    seeds = tuple(int(seed) for seed in selection["seeds"])
    if len(seeds) != 24 or len(set(seeds)) != 24:
        raise ValueError("EXP-0020 requires exactly 24 unique held-out seeds")

    parent_reports = [
        _load_report(root / "evaluation" / "parent" / mode / "report.json", seeds)
        for mode in MODES
    ]
    parent = _aggregate(parent_reports)
    trials: dict[str, dict[str, Any]] = {}
    for trial in TRIALS:
        reports = [
            _load_report(root / "evaluation" / trial / mode / "report.json", seeds)
            for mode in MODES
        ]
        trials[trial] = _aggregate(reports)

    all_trial_reports = [
        _load_report(root / "evaluation" / trial / mode / "report.json", seeds)
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
