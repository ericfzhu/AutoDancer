from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.assisted_boss_compare import (
    MODES,
    TRIAL_SEEDS,
    compare_assisted_boss,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_assisted_boss_comparator_applies_predeclared_gate_and_ranking(tmp_path) -> None:
    heldout = list(range(67001, 67013))
    expected_rates = {68001: 4, 68002: 8, 68003: 6}
    for training_seed in TRIAL_SEEDS:
        training_dir = tmp_path / "training" / f"seed-{training_seed}"
        training_dir.mkdir(parents=True)
        (training_dir / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "global_step": 51200,
                    "updates": 50,
                    "episodes": 12,
                    "curriculum_completions": expected_rates[training_seed],
                    "worker_restarts": 0,
                    "policy_loss": 0.1,
                    "value_loss": 0.2,
                    "entropy": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for directory, policy_mode, policy_seed in MODES:
            results = [
                {
                    "seed": seed,
                    "status": (
                        "curriculum_complete"
                        if index < expected_rates[training_seed]
                        else "dead"
                    ),
                    "furthest_zone": 2 if index < expected_rates[training_seed] else 1,
                    "furthest_floor": 1 if index < expected_rates[training_seed] else 4,
                    "turns": 50 + index,
                    "boss_damage": 1,
                    "boss_kills": int(index < expected_rates[training_seed]),
                }
                for index, seed in enumerate(heldout)
            ]
            write_json(
                tmp_path
                / "evaluation"
                / f"seed-{training_seed}"
                / directory
                / "report.json",
                {
                    "controller_valid": True,
                    "infrastructure_events": [],
                    "worker_restarts": 0,
                    "policy_mode": policy_mode,
                    "policy_seed": policy_seed,
                    "curriculum_start_level": 4,
                    "curriculum_target_level": 5,
                    "curriculum_profile": "boss1hp-player20",
                    "trained": {"results": results},
                },
            )

    result = compare_assisted_boss(tmp_path)
    assert result["passed"] is True
    assert result["selected_training_seed"] == 68002
    assert result["aggregate"]["successes"] == (4 + 8 + 6) * 3
    assert result["normal_start_promotable"] is False
    assert (tmp_path / "comparison.json").exists()
