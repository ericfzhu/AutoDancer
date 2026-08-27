"""Paired comparison of live recurrent-state evaluation modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from autodancer.progress import level_progress

MODES = ("carry", "reset-on-floor-transition", "reset-every-step")
MATCHED_FIELDS = (
    "checkpoint_sha256",
    "seeds",
    "policy_seed",
    "policy_mode",
    "action_contract",
    "curriculum_start_level",
    "curriculum_target_level",
    "curriculum_profile",
    "max_steps_per_episode",
)


def _load_report(path: Path, mode: str) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read recurrent-state report {path}: {error}") from error
    if not isinstance(report, dict) or int(report.get("schema_version", 0)) != 2:
        raise ValueError(f"Unexpected report schema in {path}")
    if report.get("recurrent_state_mode") != mode:
        raise ValueError(f"Expected recurrent state mode {mode!r} in {path}")
    if not report.get("controller_valid") or report.get("infrastructure_events"):
        raise ValueError(f"Controller-invalid recurrent-state report: {path}")
    trained = report.get("trained")
    if not isinstance(trained, dict) or not isinstance(trained.get("results"), list):
        raise ValueError(f"Missing trained episode results in {path}")
    return report


def _episodes(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    results = report["trained"]["results"]
    indexed = {int(result["seed"]): result for result in results}
    expected = [int(seed) for seed in report["seeds"]]
    if len(indexed) != len(results) or set(indexed) != set(expected):
        raise ValueError("Recurrent-state report has duplicate, missing, or unexpected seeds")
    return indexed


def _progress(episode: dict[str, Any]) -> int:
    return level_progress(
        int(episode.get("furthest_zone", 0)),
        int(episode.get("furthest_floor", 0)),
    )


def _summary(episodes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    values = list(episodes.values())
    progress = [_progress(episode) for episode in values]
    return {
        "episodes": len(values),
        "mean_progress": fmean(progress),
        "furthest_progress": max(progress),
        "floor2_or_better": sum(value >= 2 for value in progress),
        "floor3_or_better": sum(value >= 3 for value in progress),
        "zone2_or_better": sum(value >= 5 for value in progress),
        "death_rate": fmean(str(episode.get("status")) == "dead" for episode in values),
        "mean_turns": fmean(float(episode.get("turns", 0)) for episode in values),
    }


def _paired(
    reference: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    deltas = {
        seed: _progress(candidate[seed]) - _progress(reference[seed])
        for seed in sorted(reference)
    }
    boundary_seeds = [seed for seed, episode in reference.items() if _progress(episode) >= 2]
    return {
        "mean_progress_delta": fmean(deltas.values()),
        "improved_seeds": [seed for seed, delta in deltas.items() if delta > 0],
        "regressed_seeds": [seed for seed, delta in deltas.items() if delta < 0],
        "unchanged_seeds": sum(delta == 0 for delta in deltas.values()),
        "carry_floor2_or_better_seeds": sorted(boundary_seeds),
        "candidate_progress_on_carry_boundary_seeds": {
            str(seed): _progress(candidate[seed]) for seed in sorted(boundary_seeds)
        },
    }


def compare_recurrent_states(root: Path) -> dict[str, Any]:
    reports = {
        mode: _load_report(root / mode / "report.json", mode)
        for mode in MODES
    }
    reference = reports["carry"]
    for mode, report in reports.items():
        for field in MATCHED_FIELDS:
            if report.get(field) != reference.get(field):
                raise ValueError(f"Recurrent-state reports differ in {field!r}: {mode}")
    episodes = {mode: _episodes(report) for mode, report in reports.items()}
    return {
        "schema_version": 1,
        "valid": True,
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "seeds": reference["seeds"],
        "policy_mode": reference["policy_mode"],
        "policy_seed": reference["policy_seed"],
        "action_contract": reference["action_contract"],
        "summaries": {mode: _summary(values) for mode, values in episodes.items()},
        "paired_against_carry": {
            mode: _paired(episodes["carry"], episodes[mode])
            for mode in MODES
            if mode != "carry"
        },
        "interpretation": (
            "reset-on-floor-transition isolates cross-floor context; reset-every-step "
            "ablates all accumulated LSTM state. This diagnostic does not promote a policy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare recurrent-state ablation reports")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = compare_recurrent_states(arguments.root)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(arguments.output)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
