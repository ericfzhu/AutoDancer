"""Compare controlled learning curves and broad gameplay for Architecture 8."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ARMS = ("a2-legacy", "a2-fixed", "a8")
CURVE_STEPS = (0, 10_240, 20_480, 30_720)
FINAL_STEP = CURVE_STEPS[-1]


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _without_results(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "results"}


def _per_episode(summary: dict[str, Any], field: str) -> float:
    return float(summary.get(field, 0.0)) / max(int(summary.get("episodes", 0)), 1)


def _arm_directory(root: Path, category: str, arm: str, control_attempt: str | None) -> Path:
    base = root / category
    if control_attempt and arm in {"a2-legacy", "a2-fixed"}:
        base /= control_attempt
    return base / arm


def _training_health(
    root: Path, arm: str, control_attempt: str | None = None
) -> dict[str, Any]:
    directory = _arm_directory(root, "training", arm, control_attempt)
    final = directory / "final.pt"
    metrics_path = directory / "metrics.jsonl"
    if not final.is_file() or not metrics_path.is_file():
        return {"complete": False, "valid": False}
    payload = torch.load(final, map_location="cpu", weights_only=False)
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    losses = [
        float(row[name])
        for row in rows
        for name in ("policy_loss", "value_loss")
        if name in row
    ]
    result = {
        "complete": int(payload.get("global_step", 0)) == FINAL_STEP,
        "global_step": int(payload.get("global_step", 0)),
        "updates": int(payload.get("updates", 0)),
        "architecture": int(payload.get("architecture", {}).get("version", 0)),
        "action_contract": payload.get("checkpoint_metadata", {}).get("action_contract"),
        "finite_parameters": all(
            torch.isfinite(value).all().item() for value in payload["model"].values()
        ),
        "finite_losses": bool(losses) and all(math.isfinite(value) for value in losses),
        "worker_restarts": max((int(row.get("worker_restarts", 0)) for row in rows), default=0),
    }
    result["valid"] = (
        result["complete"]
        and result["finite_parameters"]
        and result["finite_losses"]
        and result["worker_restarts"] == 0
    )
    return result


def _curve(
    root: Path, control_attempt: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    evaluation = root / "curve-evaluation"
    for arm in ARMS:
        for step in CURVE_STEPS:
            # A8 is exactly the fixed-contract A2 policy only at step zero.
            # Every trained A8 point has its own evaluation report.
            source_arm = "a2-fixed" if arm == "a8" and step == 0 else arm
            source_directory = evaluation
            if control_attempt and source_arm in {"a2-legacy", "a2-fixed"}:
                source_directory /= control_attempt
            report = _read(source_directory / f"{source_arm}-step-{step:08d}.json")
            if report is None:
                continue
            result[arm].append(
                {
                    "step": step,
                    "source_arm": source_arm,
                    "summary": _without_results(report["trained"]),
                    "worker_restarts": int(report.get("worker_restarts", 0)),
                }
            )
    return result


def _final_curve(curves: dict[str, list[dict[str, Any]]], arm: str) -> dict[str, Any] | None:
    return next(
        (item["summary"] for item in curves[arm] if item["step"] == FINAL_STEP), None
    )


def _representation(root: Path, name: str) -> dict[str, Any] | None:
    report = _read(root / f"representation-{name}.json")
    return None if report is None else report["checkpoints"][0]


def compare_experiment(
    root: Path, *, control_attempt: str | None = None
) -> dict[str, Any]:
    curves = _curve(root, control_attempt)
    health = {
        arm: _training_health(root, arm, control_attempt)
        for arm in ARMS
    }
    warmup_representation = _representation(root, "warmup")
    final_representation = _representation(root, "final")
    fixed = _final_curve(curves, "a2-fixed")
    candidate = _final_curve(curves, "a8")
    curve_criteria = {
        "complete_curves": all(len(curves[arm]) == len(CURVE_STEPS) for arm in ARMS),
        "valid_controller_evaluations": all(
            item["worker_restarts"] == 0
            for arm in ARMS
            for item in curves[arm]
        ),
        "healthy_training": all(value.get("valid", False) for value in health.values()),
        "warmup_representation_material": bool(
            warmup_representation and warmup_representation["all_new_groups_material"]
        ),
        "final_representation_material": bool(
            final_representation and final_representation["all_new_groups_material"]
        ),
        "candidate_retains_90_percent_progress": bool(
            fixed
            and candidate
            and float(candidate["mean_progress"]) >= 0.9 * float(fixed["mean_progress"])
        ),
        "candidate_death_within_ten_points": bool(
            fixed
            and candidate
            and float(candidate["death_rate"]) <= float(fixed["death_rate"]) + 0.10
        ),
        "candidate_retains_60_percent_kills": bool(
            fixed
            and candidate
            and _per_episode(candidate, "enemy_kills")
            >= 0.6 * _per_episode(fixed, "enemy_kills")
        ),
        "candidate_retains_60_percent_items": bool(
            fixed
            and candidate
            and _per_episode(candidate, "item_pickups")
            >= 0.6 * _per_episode(fixed, "item_pickups")
        ),
        "candidate_step_limit_within_ten_points": bool(
            fixed
            and candidate
            and float(candidate["step_limit_rate"])
            <= float(fixed["step_limit_rate"]) + 0.10
        ),
    }
    curve_gate_passed = all(curve_criteria.values())

    broad_directory = root / "broad-evaluation"
    if control_attempt:
        broad_directory /= control_attempt
    broad_reports = {arm: _read(broad_directory / f"{arm}.json") for arm in ARMS}
    broad = {
        arm: None if report is None else _without_results(report["trained"])
        for arm, report in broad_reports.items()
    }
    legacy_broad, fixed_broad, candidate_broad = (
        broad["a2-legacy"],
        broad["a2-fixed"],
        broad["a8"],
    )
    broad_complete = all(value is not None for value in broad.values())
    broad_criteria = {
        "complete_reports": broad_complete,
        "candidate_progress_beats_both_controls": bool(
            broad_complete
            and float(candidate_broad["mean_progress"])
            > max(
                float(legacy_broad["mean_progress"]),
                float(fixed_broad["mean_progress"]),
            )
        ),
        "death_within_five_points_of_fixed": bool(
            broad_complete
            and float(candidate_broad["death_rate"])
            <= float(fixed_broad["death_rate"]) + 0.05
        ),
        "kills_retain_80_percent_of_fixed": bool(
            broad_complete
            and _per_episode(candidate_broad, "enemy_kills")
            >= 0.8 * _per_episode(fixed_broad, "enemy_kills")
        ),
        "items_retain_80_percent_of_fixed": bool(
            broad_complete
            and _per_episode(candidate_broad, "item_pickups")
            >= 0.8 * _per_episode(fixed_broad, "item_pickups")
        ),
        "step_limit_not_worse_than_fixed": bool(
            broad_complete
            and float(candidate_broad["step_limit_rate"])
            <= float(fixed_broad["step_limit_rate"])
        ),
        "unchanged_position_not_worse_than_fixed": bool(
            broad_complete
            and float(candidate_broad["unchanged_position_rate"])
            <= float(fixed_broad["unchanged_position_rate"])
        ),
        "no_evaluation_restarts": bool(
            broad_complete
            and all(int(report.get("worker_restarts", 0)) == 0 for report in broad_reports.values())
        ),
    }
    broad_gate_passed = curve_gate_passed and all(broad_criteria.values())
    decision = (
        "a8_ready_for_multiseed_confirmation"
        if broad_gate_passed
        else "retain_a2_after_broad_integration"
        if broad_complete
        else "ready_for_broad_gameplay"
        if curve_gate_passed
        else "stop_before_broad_gameplay"
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "training_seed": 36_001,
        "control_attempt": control_attempt or "original",
        "curve_evaluation_seeds": list(range(45_001, 45_017)),
        "broad_evaluation_seeds": list(range(46_001, 46_031)),
        "curves": curves,
        "training_health": health,
        "warmup_representation": warmup_representation,
        "final_representation": final_representation,
        "curve_criteria": curve_criteria,
        "curve_gate_passed": curve_gate_passed,
        "broad": broad,
        "broad_criteria": broad_criteria,
        "broad_gate_passed": broad_gate_passed,
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare A2/A8 controls and gameplay")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-attempt")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = compare_experiment(arguments.root, control_attempt=arguments.control_attempt)
    default_name = (
        f"comparison-{arguments.control_attempt}.json"
        if arguments.control_attempt
        else "comparison.json"
    )
    output = arguments.output or arguments.root / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "curve_gate_passed": report["curve_gate_passed"],
                "broad_gate_passed": report["broad_gate_passed"],
                "decision": report["decision"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
