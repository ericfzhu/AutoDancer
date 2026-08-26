"""Compare the predeclared EXP-0008 A2/A8 training-horizon curves."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ARMS = ("a2-frozen", "a2-continuation", "a8-continuation")
TRAINED_ARMS = ARMS[1:]
CURVE_STEPS = (30_720, 61_440, 122_880, 250_880)
FINAL_STEP = CURVE_STEPS[-1]


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {key: value for key, value in report["trained"].items() if key != "results"}


def _evaluation_valid(report: dict[str, Any] | None) -> bool:
    return bool(
        report
        and report.get("controller_valid") is True
        and int(report.get("worker_restarts", 0)) == 0
        and not report.get("infrastructure_events")
    )


def _per_episode(summary: dict[str, Any], field: str) -> float:
    return float(summary.get(field, 0.0)) / max(int(summary.get("episodes", 0)), 1)


def _training_health(root: Path, arm: str) -> dict[str, Any]:
    directory = root / "training" / arm
    paths = {
        "checkpoint": directory / "final.pt",
        "metrics": directory / "metrics.jsonl",
        "lineage": directory / "lineage.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return {"complete": False, "valid": False}
    payload = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    rows = [
        json.loads(line)
        for line in paths["metrics"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lineage = _read(paths["lineage"]) or {}
    losses = [
        float(row[name]) for row in rows for name in ("policy_loss", "value_loss") if name in row
    ]
    expected_architecture = 2 if arm == "a2-continuation" else 8
    result = {
        "complete": int(payload.get("global_step", 0)) == FINAL_STEP,
        "global_step": int(payload.get("global_step", 0)),
        "updates": int(payload.get("updates", 0)),
        "architecture": int(payload.get("architecture", {}).get("version", 0)),
        "finite_parameters": all(
            torch.isfinite(value).all().item() for value in payload["model"].values()
        ),
        "finite_losses": bool(losses) and all(math.isfinite(value) for value in losses),
        "worker_restarts": max((int(row.get("worker_restarts", 0)) for row in rows), default=0),
        "lineage_status": lineage.get("status"),
        "lineage_arm": lineage.get("experiment", {}).get("arm"),
    }
    result["valid"] = (
        result["complete"]
        and result["architecture"] == expected_architecture
        and result["finite_parameters"]
        and result["finite_losses"]
        and result["worker_restarts"] == 0
        and result["lineage_status"] == "completed"
        and result["lineage_arm"] == arm
    )
    return result


def _report_path(root: Path, arm: str, step: int) -> Path:
    if arm == "a2-frozen":
        return root / "curve-evaluation" / arm / "report.json"
    return root / "curve-evaluation" / arm / f"step-{step:08d}" / "report.json"


def compare_horizon(root: Path) -> dict[str, Any]:
    reports: dict[str, dict[int, dict[str, Any] | None]] = {
        "a2-frozen": {30_720: _read(_report_path(root, "a2-frozen", 30_720))},
        **{
            arm: {step: _read(_report_path(root, arm, step)) for step in CURVE_STEPS}
            for arm in TRAINED_ARMS
        },
    }
    curves = {
        arm: [
            {
                "step": step,
                "summary": _summary(report),
                "controller_valid": _evaluation_valid(report),
            }
            for step, report in arm_reports.items()
            if report is not None
        ]
        for arm, arm_reports in reports.items()
    }
    health = {arm: _training_health(root, arm) for arm in TRAINED_ARMS}
    representation = _read(root / "representation-final.json")
    complete = len(curves["a2-frozen"]) == 1 and all(
        len(curves[arm]) == len(CURVE_STEPS) for arm in TRAINED_ARMS
    )
    valid_evaluations = complete and all(
        point["controller_valid"] for points in curves.values() for point in points
    )

    def point(arm: str, step: int) -> dict[str, Any] | None:
        report = reports[arm].get(step)
        return _summary(report)

    frozen = point("a2-frozen", 30_720)
    a2_final = point("a2-continuation", FINAL_STEP)
    a8_final = point("a8-continuation", FINAL_STEP)
    extended_advantages = [
        step
        for step in CURVE_STEPS[1:]
        if point("a2-continuation", step)
        and point("a8-continuation", step)
        and float(point("a8-continuation", step)["mean_progress"])
        > float(point("a2-continuation", step)["mean_progress"])
    ]
    consecutive_final_advantage = (
        len(extended_advantages) >= 2
        and extended_advantages[-1] == FINAL_STEP
        and extended_advantages[-2] == CURVE_STEPS[-2]
    )
    comparable = bool(frozen and a2_final and a8_final)
    criteria = {
        "complete_curves": complete,
        "valid_controller_evaluations": valid_evaluations,
        "healthy_training": all(value.get("valid", False) for value in health.values()),
        "final_representation_material": bool(
            representation and representation["checkpoints"][0]["all_new_groups_material"]
        ),
        "two_consecutive_final_advantages": consecutive_final_advantage,
        "final_progress_beats_both_controls": bool(
            comparable
            and float(a8_final["mean_progress"])
            > max(float(frozen["mean_progress"]), float(a2_final["mean_progress"]))
        ),
        "death_within_ten_points_of_frozen": bool(
            comparable and float(a8_final["death_rate"]) <= float(frozen["death_rate"]) + 0.10
        ),
        "kills_retain_80_percent_of_frozen": bool(
            comparable
            and _per_episode(a8_final, "enemy_kills")
            >= 0.8 * _per_episode(frozen, "enemy_kills")
        ),
        "items_retain_80_percent_of_frozen": bool(
            comparable
            and _per_episode(a8_final, "item_pickups")
            >= 0.8 * _per_episode(frozen, "item_pickups")
        ),
        "step_limit_below_frozen": bool(
            comparable and float(a8_final["step_limit_rate"]) < float(frozen["step_limit_rate"])
        ),
    }
    evidence_valid = all(
        criteria[name]
        for name in (
            "complete_curves",
            "valid_controller_evaluations",
            "healthy_training",
            "final_representation_material",
        )
    )
    passed = all(criteria.values())
    final_only = bool(
        not criteria["two_consecutive_final_advantages"]
        and all(
            value
            for name, value in criteria.items()
            if name != "two_consecutive_final_advantages"
        )
    )
    decision = (
        "a8_ready_for_multiseed_confirmation"
        if passed
        else "a8_final_only_advantage_requires_replication"
        if final_only
        else "retain_a2_after_long_horizon"
        if evidence_valid
        else "invalid_or_incomplete_horizon_evidence"
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": "EXP-0008",
        "training_seed": 51_001,
        "evaluation_seeds": list(range(56_001, 56_017)),
        "curve_steps": list(CURVE_STEPS),
        "curves": curves,
        "training_health": health,
        "final_representation": representation,
        "extended_advantage_steps": extended_advantages,
        "criteria": criteria,
        "passed": passed,
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the EXP-0008 A8 horizon test")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = compare_horizon(arguments.root)
    output = arguments.output or arguments.root / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "passed": report["passed"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
