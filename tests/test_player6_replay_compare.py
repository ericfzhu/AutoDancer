from __future__ import annotations

import json

from autodancer.training.player6_replay_compare import (
    CANDIDATES,
    MODES,
    PROFILES,
    TRAINED,
    compare_player6_replay,
)


def test_player6_replay_comparison_applies_improvement_retention_and_health(tmp_path) -> None:
    seeds = list(range(74001, 74025))
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "heldout-selection.json").write_text(
        json.dumps({"seeds": seeds}), encoding="utf-8"
    )
    player6_successes = {
        "parent": 10,
        "seed-73001": 15,
        "seed-73002": 14,
        "seed-73003": 9,
    }
    for candidate in CANDIDATES:
        for profile in PROFILES:
            for name, mode, policy_seed in MODES:
                directory = evaluation / candidate / profile / name
                directory.mkdir(parents=True)
                if mode == "deterministic":
                    successes = 6
                elif profile == "boss1hp-player6":
                    successes = player6_successes[candidate]
                else:
                    successes = 18
                results = [
                    {
                        "seed": seed,
                        "status": "curriculum_complete" if index < successes else "dead",
                        "furthest_zone": 2 if index < successes else 1,
                        "turns": 40 + index,
                    }
                    for index, seed in enumerate(seeds)
                ]
                (directory / "report.json").write_text(
                    json.dumps(
                        {
                            "controller_valid": True,
                            "infrastructure_events": [],
                            "worker_restarts": 0,
                            "policy_mode": mode,
                            "policy_seed": policy_seed,
                            "curriculum_profile": profile,
                            "trained": {"results": results},
                        }
                    ),
                    encoding="utf-8",
                )
    for candidate in TRAINED:
        directory = tmp_path / "training" / candidate
        directory.mkdir(parents=True)
        (directory / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "global_step": 51200,
                    "updates": 50,
                    "policy_loss": 0.1,
                    "value_loss": 0.2,
                    "entropy": 1.0,
                    "gradient_norm_preclip": 0.5,
                    "worker_restarts": 0,
                    "collector_recoveries_total": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    result = compare_player6_replay(tmp_path)
    assert result["passed"] is True
    assert result["improved_trials"] == ["seed-73001", "seed-73002"]
    assert result["selected_trial"] == "seed-73001"
    assert result["decision"] == "advance_to_boss_health"
