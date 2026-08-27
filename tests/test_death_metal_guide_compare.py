from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.death_metal_guide_compare import MODES, TRIALS, compare


def _report(seeds: list[int], phase4: set[int], *, damage: int) -> dict[str, object]:
    return {
        "controller_valid": True,
        "worker_restarts": 0,
        "infrastructure_events": [],
        "seeds": seeds,
        "curriculum_profile": "player20",
        "curriculum_start_level": 4,
        "curriculum_target_level": 5,
        "trained": {
            "results": [
                {
                    "seed": seed,
                    "status": "curriculum_complete" if seed in phase4 else "dead",
                    "death_metal_phase4_reached": seed in phase4,
                    "boss_phase_depth": 4 if seed in phase4 else 2,
                    "boss_damage": damage,
                    "turns": 100,
                }
                for seed in seeds
            ]
        },
    }


def test_guide_gate_requires_reproducible_phase_acquisition(tmp_path: Path) -> None:
    seeds = list(range(81001, 81025))
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "heldout-selection.json").write_text(
        json.dumps({"seeds": seeds}), encoding="utf-8"
    )
    parent_phase4 = set(seeds[:2])
    trained_phase4 = set(seeds[:12])
    for name, phases, damage in (
        ("parent", parent_phase4, 1),
        *((trial, trained_phase4, 3) for trial in TRIALS),
    ):
        for mode in MODES:
            directory = evaluation / name / mode
            directory.mkdir(parents=True)
            (directory / "report.json").write_text(
                json.dumps(_report(seeds, phases, damage=damage)), encoding="utf-8"
            )

    result = compare(tmp_path)

    assert result["gate"]["passed"] is True
    assert result["selected_trial"] in TRIALS
    assert result["aggregate"]["phase4_rate"] == 0.5
    assert len(result["aggregate"]["distinct_phase4_seeds"]) == 12


def test_guide_gate_rejects_one_lucky_checkpoint(tmp_path: Path) -> None:
    seeds = list(range(81001, 81025))
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "heldout-selection.json").write_text(
        json.dumps({"seeds": seeds}), encoding="utf-8"
    )
    for name in ("parent", *TRIALS):
        phases = set(seeds[:12]) if name == TRIALS[0] else set()
        for mode in MODES:
            directory = evaluation / name / mode
            directory.mkdir(parents=True)
            (directory / "report.json").write_text(
                json.dumps(_report(seeds, phases, damage=3)), encoding="utf-8"
            )

    result = compare(tmp_path)

    assert result["gate"]["passed"] is False
    assert result["selected_trial"] is None


def test_exp0020_launcher_uses_only_legal_player_health_assistance() -> None:
    source = Path("tools/run-exp0020-legal-death-metal-guide.ps1").read_text(
        encoding="utf-8"
    )

    assert "boss1hp" not in source
    assert '"--curriculum-profile", "player20"' in source
    assert '"--initialize-from", $guideSource' in source
    assert '"--freeze-base-updates", "10"' in source
    assert "controller-qualification-player-health-only-world-ready" in source
    assert "$guideTotalSteps = 122880" in source
