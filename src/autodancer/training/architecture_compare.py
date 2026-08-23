"""Compare schema-9 architecture pilots using gameplay outcomes only."""

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


def _without_results(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "results"}


def _rank(summary: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    episodes = max(int(summary["episodes"]), 1)
    return (
        float(summary["mean_progress"]),
        -float(summary["death_rate"]),
        -float(summary["step_limit_rate"]),
        float(summary["enemy_kills"]) / episodes,
        float(summary["item_pickups"]) / episodes,
        float(summary["unique_positions_per_100_turns"]),
    )


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
    parameters = payload["model"].values()
    result = {
        "global_step": int(payload.get("global_step", 0)),
        "updates": int(payload.get("updates", 0)),
        "architecture_version": payload.get("architecture", {}).get("version"),
        "reward_version": payload.get("checkpoint_metadata", {})
        .get("reward", {})
        .get("version"),
        "finite_parameters": all(torch.isfinite(value).all().item() for value in parameters),
        "finite_losses": bool(losses) and all(math.isfinite(value) for value in losses),
    }
    result["valid"] = (
        result["global_step"] == PILOT_STEPS
        and result["updates"] == PILOT_UPDATES
        and result["architecture_version"] == 6
        and result["reward_version"] == 2
        and result["finite_parameters"]
        and result["finite_losses"]
    )
    return result


def compare_experiment(root: Path) -> dict[str, Any]:
    evaluation = root / "evaluation"
    reference_report = json.loads((evaluation / "v2-final.json").read_text(encoding="utf-8"))
    reference = reference_report["trained"]
    candidates = []
    reports = []
    for seed in (33001, 33002, 33003):
        report = json.loads(
            (evaluation / f"arch6-seed-{seed}.json").read_text(encoding="utf-8")
        )
        reports.append(report)
        summary = report["trained"]
        training = _validate_training(root / "training" / f"seed-{seed}")
        candidates.append(
            {
                "seed": seed,
                "summary": _without_results(summary),
                "training": training,
                "evaluation_worker_restarts": int(report.get("worker_restarts", 0)),
            }
        )

    aggregate = summarize_episodes(
        [episode for report in reports for episode in report["trained"]["results"]],
        "architecture_6",
    )
    progress_directions = sum(
        float(candidate["summary"]["mean_progress"]) > float(reference["mean_progress"])
        for candidate in candidates
    )
    criteria = {
        "aggregate_progress_improved": (
            float(aggregate["mean_progress"]) > float(reference["mean_progress"])
        ),
        "progress_improved_in_two_pilots": progress_directions >= 2,
        "all_training_valid": all(candidate["training"]["valid"] for candidate in candidates),
        "no_evaluation_restarts": all(
            candidate["evaluation_worker_restarts"] == 0 for candidate in candidates
        )
        and int(reference_report.get("worker_restarts", 0)) == 0,
    }
    passed = all(criteria.values())
    selected = (
        max(candidates, key=lambda candidate: _rank(candidate["summary"])) if passed else None
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "reference": _without_results(reference),
        "aggregate": _without_results(aggregate),
        "candidates": candidates,
        "criteria": criteria,
        "passed": passed,
        "selected_seed": None if selected is None else selected["seed"],
        "decision": "continue_architecture_6" if passed else "retain_v2_architecture_2",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare architecture-6 live-game pilots")
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
