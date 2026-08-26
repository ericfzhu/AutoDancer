"""Evaluate the predeclared qualified A8 replication against A2 controls."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ARMS = ("a2-frozen", "a2-finetune", "a8-candidate")
TRAINED_ARMS = ARMS[1:]
CURVE_STEPS = (0, 10_240, 20_480, 30_720)
FINAL_STEP = CURVE_STEPS[-1]


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {key: value for key, value in report["trained"].items() if key != "results"}


def _per_episode(summary: dict[str, Any], field: str) -> float:
    return float(summary.get(field, 0.0)) / max(int(summary.get("episodes", 0)), 1)


def _evaluation_valid(report: dict[str, Any] | None) -> bool:
    return bool(
        report
        and report.get("controller_valid") is True
        and int(report.get("worker_restarts", 0)) == 0
        and not report.get("infrastructure_events")
    )


def _training_health(root: Path, arm: str) -> dict[str, Any]:
    directory = root / "training" / arm
    final = directory / "final.pt"
    metrics_path = directory / "metrics.jsonl"
    lineage_path = directory / "lineage.json"
    if not final.is_file() or not metrics_path.is_file() or not lineage_path.is_file():
        return {"complete": False, "valid": False}
    payload = torch.load(final, map_location="cpu", weights_only=False)
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    losses = [
        float(row[name]) for row in rows for name in ("policy_loss", "value_loss") if name in row
    ]
    expected_architecture = 2 if arm == "a2-finetune" else 8
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
        "lineage_status": lineage.get("status"),
        "lineage_arm": lineage.get("experiment", {}).get("arm"),
    }
    result["valid"] = (
        result["complete"]
        and result["architecture"] == expected_architecture
        and result["action_contract"] == "current"
        and result["finite_parameters"]
        and result["finite_losses"]
        and result["worker_restarts"] == 0
        and result["lineage_status"] == "completed"
        and result["lineage_arm"] == arm
    )
    return result


def _curve_report(root: Path, arm: str, step: int) -> dict[str, Any] | None:
    if step == 0:
        path = root / "curve-evaluation" / "a2-frozen" / "report.json"
    else:
        path = root / "curve-evaluation" / arm / f"step-{step:08d}" / "report.json"
    return _read(path)


def _curve(root: Path) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        steps = (0,) if arm == "a2-frozen" else CURVE_STEPS
        for step in steps:
            report = _curve_report(root, arm, step)
            if report is not None:
                curves[arm].append(
                    {
                        "step": step,
                        "source_arm": "a2-frozen" if step == 0 else arm,
                        "summary": _summary(report),
                        "controller_valid": _evaluation_valid(report),
                    }
                )
    return curves


def _curve_final(curves: dict[str, list[dict[str, Any]]], arm: str) -> dict[str, Any] | None:
    return next(
        (point["summary"] for point in curves[arm] if point["step"] == FINAL_STEP),
        None,
    )


def compare_replication(
    root: Path,
    *,
    training_seed: int = 51_001,
    curve_seeds: list[int] | None = None,
    broad_seeds: list[int] | None = None,
) -> dict[str, Any]:
    curve_seeds = list(range(52_001, 52_017)) if curve_seeds is None else curve_seeds
    broad_seeds = list(range(53_001, 53_031)) if broad_seeds is None else broad_seeds
    parity = _read(root / "parity.json")
    warmup = _read(root / "representation-warmup.json")
    final = _read(root / "representation-final.json")
    curves = _curve(root)
    health = {arm: _training_health(root, arm) for arm in TRAINED_ARMS}
    finetune = _curve_final(curves, "a2-finetune")
    candidate = _curve_final(curves, "a8-candidate")
    expected_curve_points = {
        "a2-frozen": 1,
        "a2-finetune": len(CURVE_STEPS),
        "a8-candidate": len(CURVE_STEPS),
    }
    curve_criteria = {
        "parity_passed": bool(parity and parity.get("passed")),
        "complete_curves": all(len(curves[arm]) == expected_curve_points[arm] for arm in ARMS),
        "valid_controller_evaluations": all(
            point["controller_valid"] for arm in ARMS for point in curves[arm]
        ),
        "healthy_training": all(value.get("valid", False) for value in health.values()),
        "warmup_representation_material": bool(
            warmup and warmup["checkpoints"][0]["all_new_groups_material"]
        ),
        "final_representation_material": bool(
            final and final["checkpoints"][0]["all_new_groups_material"]
        ),
        "candidate_retains_90_percent_progress": bool(
            finetune
            and candidate
            and float(candidate["mean_progress"]) >= 0.9 * float(finetune["mean_progress"])
        ),
        "candidate_death_within_ten_points": bool(
            finetune
            and candidate
            and float(candidate["death_rate"]) <= float(finetune["death_rate"]) + 0.10
        ),
        "candidate_retains_60_percent_kills": bool(
            finetune
            and candidate
            and _per_episode(candidate, "enemy_kills")
            >= 0.6 * _per_episode(finetune, "enemy_kills")
        ),
        "candidate_retains_60_percent_items": bool(
            finetune
            and candidate
            and _per_episode(candidate, "item_pickups")
            >= 0.6 * _per_episode(finetune, "item_pickups")
        ),
        "candidate_step_limit_within_ten_points": bool(
            finetune
            and candidate
            and float(candidate["step_limit_rate"]) <= float(finetune["step_limit_rate"]) + 0.10
        ),
    }
    curve_gate_passed = all(curve_criteria.values())

    broad_reports = {arm: _read(root / "broad-evaluation" / arm / "report.json") for arm in ARMS}
    broad = {arm: _summary(report) for arm, report in broad_reports.items()}
    frozen, finetuned, candidate_broad = (
        broad["a2-frozen"],
        broad["a2-finetune"],
        broad["a8-candidate"],
    )
    broad_complete = all(value is not None for value in broad.values())
    broad_criteria = {
        "complete_reports": broad_complete,
        "candidate_progress_beats_both_controls": bool(
            broad_complete
            and float(candidate_broad["mean_progress"])
            > max(float(frozen["mean_progress"]), float(finetuned["mean_progress"]))
        ),
        "death_within_five_points_of_finetune": bool(
            broad_complete
            and float(candidate_broad["death_rate"]) <= float(finetuned["death_rate"]) + 0.05
        ),
        "kills_retain_80_percent_of_finetune": bool(
            broad_complete
            and _per_episode(candidate_broad, "enemy_kills")
            >= 0.8 * _per_episode(finetuned, "enemy_kills")
        ),
        "items_retain_80_percent_of_finetune": bool(
            broad_complete
            and _per_episode(candidate_broad, "item_pickups")
            >= 0.8 * _per_episode(finetuned, "item_pickups")
        ),
        "step_limit_not_worse_than_finetune": bool(
            broad_complete
            and float(candidate_broad["step_limit_rate"]) <= float(finetuned["step_limit_rate"])
        ),
        "unchanged_position_not_worse_than_finetune": bool(
            broad_complete
            and float(candidate_broad["unchanged_position_rate"])
            <= float(finetuned["unchanged_position_rate"])
        ),
        "valid_controller_evaluations": bool(
            broad_complete and all(_evaluation_valid(report) for report in broad_reports.values())
        ),
    }
    broad_gate_passed = curve_gate_passed and all(broad_criteria.values())
    decision = (
        "a8_ready_for_multiseed_confirmation"
        if broad_gate_passed
        else "retain_a2_after_qualified_replication"
        if broad_complete
        else "ready_for_broad_gameplay"
        if curve_gate_passed
        else "stop_before_broad_gameplay"
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": "EXP-0007",
        "training_seed": training_seed,
        "curve_evaluation_seeds": curve_seeds,
        "broad_evaluation_seeds": broad_seeds,
        "curves": curves,
        "training_health": health,
        "parity": parity,
        "warmup_representation": warmup,
        "final_representation": final,
        "curve_criteria": curve_criteria,
        "curve_gate_passed": curve_gate_passed,
        "broad": broad,
        "broad_criteria": broad_criteria,
        "broad_gate_passed": broad_gate_passed,
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the qualified A8 replication")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, default=51_001)
    parser.add_argument("--curve-seeds", default="52001-52016")
    parser.add_argument("--broad-seeds", default="53001-53030")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    def parse_range(value: str) -> list[int]:
        start, end = (int(item) for item in value.split("-", 1))
        if start < 0 or end < start:
            raise ValueError(f"Invalid seed range {value!r}")
        return list(range(start, end + 1))

    report = compare_replication(
        arguments.root,
        training_seed=arguments.training_seed,
        curve_seeds=parse_range(arguments.curve_seeds),
        broad_seeds=parse_range(arguments.broad_seeds),
    )
    output = arguments.output or arguments.root / "comparison.json"
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
