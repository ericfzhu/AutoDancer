"""Apply the predeclared gameplay gates to the paired A2/A7 pilot."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from autodancer.training.baseline import summarize_episodes

PILOT_STEPS = 51_200
PILOT_UPDATES = 50
TRAINING_SEEDS = (35_001, 35_002, 35_003)


def _without_results(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "results"}


def _per_episode(summary: dict[str, Any], field: str) -> float:
    return float(summary[field]) / max(int(summary["episodes"]), 1)


def _validate_training(run_dir: Path) -> dict[str, Any]:
    payload = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    losses = [
        float(row[name])
        for row in metrics
        for name in ("policy_loss", "value_loss")
        if name in row
    ]
    gates = [float(row["adapter_gate"]) for row in metrics if "adapter_gate" in row]
    norms = [
        float(row["adapter_parameter_norm"])
        for row in metrics
        if "adapter_parameter_norm" in row
    ]
    result = {
        "global_step": int(payload.get("global_step", 0)),
        "updates": int(payload.get("updates", 0)),
        "architecture_version": payload.get("architecture", {}).get("version"),
        "reward_version": payload.get("checkpoint_metadata", {}).get("reward", {}).get("version"),
        "finite_parameters": all(
            torch.isfinite(value).all().item() for value in payload["model"].values()
        ),
        "finite_losses": bool(losses) and all(math.isfinite(value) for value in losses),
        "gate_changed": bool(gates) and any(abs(value) > 1.0e-8 for value in gates),
        "gate_finite_and_bounded": bool(gates)
        and all(math.isfinite(value) and abs(value) < 1.0 for value in gates),
        "adapter_changed_after_gate": len(norms) > 1
        and not math.isclose(norms[0], norms[-1], rel_tol=0.0, abs_tol=1.0e-8),
    }
    result["valid"] = (
        result["global_step"] == PILOT_STEPS
        and result["updates"] == PILOT_UPDATES
        and result["architecture_version"] == 7
        and result["reward_version"] == 2
        and result["finite_parameters"]
        and result["finite_losses"]
        and result["gate_changed"]
        and result["gate_finite_and_bounded"]
        and result["adapter_changed_after_gate"]
    )
    return result


def compare_experiment(root: Path) -> dict[str, Any]:
    parity = json.loads((root / "parity.json").read_text(encoding="utf-8"))
    evaluation = root / "evaluation"
    reference_report = json.loads((evaluation / "v2-final.json").read_text(encoding="utf-8"))
    reference = reference_report["trained"]
    candidates: list[dict[str, Any]] = []
    reports = []
    for seed in TRAINING_SEEDS:
        report = json.loads((evaluation / f"arch7-seed-{seed}.json").read_text(encoding="utf-8"))
        reports.append(report)
        summary = report["trained"]
        candidates.append(
            {
                "seed": seed,
                "summary": _without_results(summary),
                "training": _validate_training(root / "training" / f"seed-{seed}"),
                "evaluation_worker_restarts": int(report.get("worker_restarts", 0)),
            }
        )
    aggregate = summarize_episodes(
        [episode for report in reports for episode in report["trained"]["results"]],
        "architecture_7",
    )
    ref_direction = float(reference.get("unchanged_direction_rate", 0.0))
    staircase_improved = (
        float(aggregate["staircase_discovery_rate"])
        > float(reference["staircase_discovery_rate"])
        or float(aggregate["staircase_conversion_rate"])
        > float(reference["staircase_conversion_rate"])
        or int(aggregate["staircase_exits"]) > int(reference["staircase_exits"])
    )
    criteria = {
        "preflight_parity": bool(parity.get("passed")),
        "aggregate_progress_improved": float(aggregate["mean_progress"])
        > float(reference["mean_progress"]),
        "progress_improved_in_two_pilots": sum(
            float(candidate["summary"]["mean_progress"]) > float(reference["mean_progress"])
            for candidate in candidates
        )
        >= 2,
        "staircase_evidence_improved": staircase_improved
        and int(aggregate["staircase_discoveries"]) > 0,
        "death_within_five_points": float(aggregate["death_rate"])
        <= float(reference["death_rate"]) + 0.05,
        "kills_retained": _per_episode(aggregate, "enemy_kills")
        >= 0.8 * _per_episode(reference, "enemy_kills"),
        "items_retained": _per_episode(aggregate, "item_pickups")
        >= 0.8 * _per_episode(reference, "item_pickups"),
        "unchanged_directions_reduced_20_percent": float(aggregate["unchanged_direction_rate"])
        <= 0.8 * ref_direction,
        "step_limit_not_worse": float(aggregate["step_limit_rate"])
        <= float(reference["step_limit_rate"]),
        "stationary_streak_not_worse": float(aggregate["mean_max_unchanged_position_streak"])
        <= float(reference["mean_max_unchanged_position_streak"]),
        "all_training_valid": all(candidate["training"]["valid"] for candidate in candidates),
        "no_evaluation_restarts": all(
            candidate["evaluation_worker_restarts"] == 0 for candidate in candidates
        )
        and int(reference_report.get("worker_restarts", 0)) == 0,
    }
    passed = all(criteria.values())
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "reference": _without_results(reference),
        "aggregate": _without_results(aggregate),
        "candidates": candidates,
        "criteria": criteria,
        "passed": passed,
        "decision": "continue_architecture_7" if passed else "retain_v2_architecture_2",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Architecture-7 live-game pilots")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = compare_experiment(arguments.root)
    output = arguments.output or arguments.root / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"decision": report["decision"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
