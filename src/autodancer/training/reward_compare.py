"""Aggregate V4 reward-arm evaluations using gameplay-only decision rules."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from autodancer.training.baseline import summarize_episodes


def _without_results(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "results"}


def aggregate_reports(reports: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    episodes = [episode for report in reports for episode in report["trained"]["results"]]
    return summarize_episodes(episodes, policy)


def evaluate_arm(
    reference: dict[str, Any],
    reports: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    aggregate = aggregate_reports(reports, name)
    episodes = int(aggregate["episodes"])
    reference_episodes = int(reference["episodes"])
    progress_directions = sum(
        float(report["trained"]["mean_progress"]) > float(reference["mean_progress"])
        for report in reports
    )
    criteria = {
        "progress_improved": aggregate["mean_progress"] > reference["mean_progress"],
        "death_rate_at_most_50_percent": aggregate["death_rate"] <= 0.50,
        "kills_retained_at_least_80_percent": (
            aggregate["enemy_kills"] / episodes
            >= 0.8 * float(reference["enemy_kills"]) / reference_episodes
        ),
        "items_retained_at_least_80_percent": (
            aggregate["item_pickups"] / episodes
            >= 0.8 * float(reference["item_pickups"]) / reference_episodes
        ),
        "idle_or_step_limit_reduced": (
            aggregate["step_limit_rate"] < reference["step_limit_rate"]
            or aggregate["idle_rate"] < reference["idle_rate"]
        ),
        "progress_improved_in_two_checkpoints": progress_directions >= 2,
    }
    return {
        "name": name,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "progress_improved_checkpoints": progress_directions,
        "aggregate": _without_results(aggregate),
        "checkpoints": [_without_results(report["trained"]) for report in reports],
    }


def choose_arm(reference: dict[str, Any], arms: list[dict[str, Any]]) -> str | None:
    passing = [arm for arm in arms if arm["passed"]]
    if not passing:
        return None

    def rank(arm: dict[str, Any]) -> tuple[float, float, float, float, float]:
        summary = arm["aggregate"]
        episodes = max(int(summary["episodes"]), 1)
        return (
            float(summary["mean_progress"]),
            -float(summary["death_rate"]),
            -float(summary["idle_rate"]),
            float(summary["enemy_kills"]) / episodes,
            float(summary["item_pickups"]) / episodes,
        )

    return str(max(passing, key=rank)["name"])


def validate_training_run(run_dir: Path, source_model: dict[str, torch.Tensor]) -> dict[str, Any]:
    payload = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    model = payload["model"]
    finite_parameters = all(torch.isfinite(value).all().item() for value in model.values())
    policy_changed = any(
        not name.startswith("critic.") and not torch.equal(value, source_model[name])
        for name, value in model.items()
    )
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loss_values = [
        float(row[name]) for row in metrics for name in ("policy_loss", "value_loss") if name in row
    ]
    result = {
        "global_step": int(payload.get("global_step", 0)),
        "updates": int(payload.get("updates", 0)),
        "finite_parameters": finite_parameters,
        "finite_losses": bool(loss_values) and all(math.isfinite(value) for value in loss_values),
        "policy_parameters_changed": policy_changed,
        "reward": payload.get("checkpoint_metadata", {}).get("reward"),
    }
    result["valid"] = (
        result["global_step"] == 51_200
        and result["updates"] == 50
        and result["finite_parameters"]
        and result["finite_losses"]
        and result["policy_parameters_changed"]
    )
    return result


def compare_experiment(root: Path) -> dict[str, Any]:
    evaluation = root / "evaluation"
    reference_report = json.loads((evaluation / "v2-final.json").read_text(encoding="utf-8"))
    reference = reference_report["trained"]
    source_payload = torch.load(
        root.parent / "reward-v2-250k" / "final.pt", map_location="cpu", weights_only=False
    )
    source_model = source_payload["model"]
    arms = []
    for arm in ("v4a", "v4b"):
        reports = [
            json.loads((evaluation / f"{arm}-seed-{seed}.json").read_text(encoding="utf-8"))
            for seed in (32001, 32002, 32003)
        ]
        arm_result = evaluate_arm(reference, reports, arm)
        training = [
            validate_training_run(root / "training" / arm / f"seed-{seed}", source_model)
            for seed in (32001, 32002, 32003)
        ]
        expected_potential = 0.5 if arm == "v4a" else 1.0
        metadata_valid = all(
            result.get("reward", {}).get("version") == 4
            and result.get("reward", {}).get("weights", {}).get("stair_potential_max")
            == expected_potential
            for result in training
        )
        arm_result["training"] = training
        arm_result["criteria"]["training_valid"] = (
            all(result["valid"] for result in training) and metadata_valid
        )
        arm_result["passed"] = all(arm_result["criteria"].values())
        arms.append(arm_result)
    selected = choose_arm(reference, arms)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "reference": _without_results(reference),
        "arms": arms,
        "selected_arm": selected,
        "decision": f"continue_{selected}" if selected else "retain_v2_reject_v4",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply V4 reward-arm decision rules")
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
