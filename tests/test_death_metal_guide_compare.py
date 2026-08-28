from __future__ import annotations

import json
from pathlib import Path

from autodancer.training.death_metal_guide_compare import MODES, TRIALS, compare


def _report(
    seeds: list[int], phase4: set[int], *, damage: int, mode: str
) -> dict[str, object]:
    policy_mode = "deterministic" if mode == "deterministic" else "stochastic"
    policy_seed = 0 if mode == "deterministic" else int(mode.removeprefix("stochastic-"))
    return {
        "controller_valid": True,
        "worker_restarts": 0,
        "infrastructure_events": [],
        "seeds": seeds,
        "policy_mode": policy_mode,
        "policy_seed": policy_seed,
        "character": "Bard",
        "action_contract": "map-navigation-prior-v1",
        "num_instances": 8,
        "max_steps_per_episode": 500,
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
                    "boss_actor_types": [101, 102, 103, 104]
                    if seed in phase4
                    else [101, 102],
                    "boss_damage": max(damage, 7) if seed in phase4 else damage,
                    "turns": 100,
                    "boss_type": 2,
                    "furthest_zone": 2 if seed in phase4 else 1,
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
                json.dumps(_report(seeds, phases, damage=damage, mode=mode)),
                encoding="utf-8",
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
                json.dumps(_report(seeds, phases, damage=3, mode=mode)), encoding="utf-8"
            )

    result = compare(tmp_path)

    assert result["gate"]["passed"] is False
    assert result["selected_trial"] is None


def test_guide_gate_rejects_inconsistent_phase_evidence(tmp_path: Path) -> None:
    seeds = list(range(81001, 81025))
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "heldout-selection.json").write_text(
        json.dumps({"seeds": seeds}), encoding="utf-8"
    )
    for name in ("parent", *TRIALS):
        for mode in MODES:
            directory = evaluation / name / mode
            directory.mkdir(parents=True)
            report = _report(seeds, set(), damage=3, mode=mode)
            if name == TRIALS[0] and mode == "deterministic":
                report["trained"]["results"][0]["death_metal_phase4_reached"] = True
            (directory / "report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

    try:
        compare(tmp_path)
    except ValueError as error:
        assert "inconsistent phase-4 evidence" in str(error)
    else:
        raise AssertionError("malformed phase evidence was accepted")


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
    assert "-not (Test-Path $guideQualification)" in source
