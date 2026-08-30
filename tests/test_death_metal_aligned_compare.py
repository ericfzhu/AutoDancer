from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from autodancer.rewards import load_reward_config
from autodancer.training.death_metal_aligned_compare import _aggregate


def test_outcome_aligned_guide_changes_only_conflicting_survival_terms() -> None:
    previous = asdict(load_reward_config("configs/reward-death-metal-guide-v2.json"))
    aligned = asdict(load_reward_config("configs/reward-death-metal-guide-v3.json"))

    changed = {name for name in previous if previous[name] != aligned[name]}

    assert changed == {"player_damage", "death"}
    assert previous["player_damage"] == -0.05
    assert aligned["player_damage"] == 0.0
    assert previous["death"] == -2.0
    assert aligned["death"] == 0.0
    assert aligned["combat_reward_scope"] == "boss_only"
    assert aligned["enemy_damage"] == 0.2


def test_aligned_aggregate_exposes_zero_contact_timeout_and_actions() -> None:
    report = {
        "trained": {
            "results": [
                {
                    "seed": 1,
                    "status": "time_limit",
                    "death_metal_phase4_reached": False,
                    "boss_phase_depth": 1,
                    "boss_damage": 0,
                    "turns": 500,
                    "action_counts": [10, 10, 10, 10, 460, 0, 0, 0, 0, 0, 0],
                },
                {
                    "seed": 2,
                    "status": "dead",
                    "death_metal_phase4_reached": True,
                    "boss_phase_depth": 4,
                    "boss_damage": 8,
                    "turns": 100,
                    "action_counts": [25, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0],
                },
            ]
        }
    }

    result = _aggregate([report])

    assert result["phase4_rate"] == 0.5
    assert result["zero_contact_timeout_rate"] == 0.5
    assert result["step_limit_rate"] == 0.5
    assert result["death_rate"] == 0.5
    assert result["boss_damage"] == 8
    assert result["wait_rate"] == 460 / 600
    assert result["movement_rate"] == 140 / 600


def test_exp0022_contract_and_launcher_pin_aligned_reward() -> None:
    contract = Path("experiments/EXP-0022/experiment.yaml").read_text(encoding="utf-8")
    launcher = Path("tools/run-exp0020-legal-death-metal-guide.ps1").read_text(encoding="utf-8")
    wrapper = Path("tools/run-exp0022-outcome-aligned-death-metal-guide.ps1").read_text(
        encoding="utf-8"
    )
    reward = json.loads(
        Path("configs/reward-death-metal-guide-v3.json").read_text(encoding="utf-8")
    )

    assert "EXP-0022" in contract
    assert "reward-death-metal-guide-v3.json" in contract
    assert "84001-84256" in contract
    assert "85001-85256" in contract
    assert "[86001, 86002, 86003]" in contract
    assert "stochastic-87001" in contract
    assert "stochastic-87002" in contract
    assert 'if ($guideExperimentId -eq "EXP-0022")' in launcher
    assert '"DeathMetalGuideV3"' in launcher
    assert '"a8-player20-outcome-aligned-guide"' in launcher
    assert '-ExperimentId "EXP-0022"' in wrapper
    assert reward["player_damage"] == 0.0
    assert reward["death"] == 0.0
    assert reward["aborted"] == -1.0
