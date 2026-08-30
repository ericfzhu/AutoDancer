from __future__ import annotations

from pathlib import Path

from autodancer.training.death_metal_full_boss_compare import _aggregate


def _episode(seed: int, *, completed: bool, phase: int, health: int) -> dict:
    return {
        "seed": seed,
        "status": "curriculum_complete" if completed else "dead",
        "boss_phase_depth": phase,
        "initial_boss_health": 9,
        "minimum_boss_health": health,
        "turns": 50,
    }


def test_full_boss_aggregate_selects_gameplay_outcomes_not_shaped_return() -> None:
    report = {
        "trained": {
            "results": [
                _episode(1, completed=True, phase=4, health=1),
                _episode(2, completed=False, phase=3, health=4),
                _episode(3, completed=False, phase=2, health=6),
                _episode(4, completed=False, phase=1, health=9),
            ]
        }
    }

    result = _aggregate([report])

    assert result["completion_rate"] == 0.25
    assert result["distinct_completion_seeds"] == [1]
    assert result["mean_phase_depth"] == 2.5
    assert result["mean_observed_boss_health_loss"] == 4


def test_exp0024_launcher_uses_every_full_boss_action_as_learner_data() -> None:
    source = Path("tools/run-exp0020-legal-death-metal-guide.ps1").read_text(
        encoding="utf-8"
    )
    wrapper = Path("tools/run-exp0024-full-boss-potential.ps1").read_text(
        encoding="utf-8"
    )

    assert '"EXP-0024"' in source and '"EXP-0024"' in wrapper
    assert "legal-death-metal-full-boss-potential" in source
    assert '"a8-player20-full-boss-potential"' in source
    assert "reward-death-metal-potential-v5.json" in source
    assert "$guideNaturalPrefix = $false" in source
