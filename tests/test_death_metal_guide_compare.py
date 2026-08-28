from __future__ import annotations

import json
from pathlib import Path

from autodancer.experiments.provenance import sha256_file
from autodancer.training.death_metal_guide_compare import (
    MODES,
    SOURCE_SHA256,
    TRIALS,
    compare,
)

TRAINING_SEEDS = list(range(80001, 80049))


def _write_selections(root: Path, seeds: list[int]) -> Path:
    reset = {
        "id": "boss-identity",
        "start_level": 4,
        "target_level": 5,
        "profile": "player20",
    }
    calibration = root / "calibration"
    calibration.mkdir()
    reports: dict[str, tuple[Path, list[int]]] = {}
    for bank, candidates in (
        ("training", list(range(80001, 80257))),
        ("evaluation", list(range(81001, 81257))),
    ):
        path = calibration / f"{bank}-candidates.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "boss-identity-calibration-v1",
                    "protocol_schema_version": 10,
                    "game_version": "v4.2.1-b5713",
                    "steam_build": "22938426",
                    "character": "Bard",
                    "mode": "AllZonesSeededCurriculum",
                    "num_instances": 8,
                    "curriculum_reset": reset,
                    "controller_valid": True,
                    "worker_restarts": 0,
                    "infrastructure_events": [],
                    "seeds": candidates,
                    "disclosure": (
                        "reset boss identity only; no gameplay action was selected or issued"
                    ),
                    "controller_qualification_sha256": "a" * 64,
                    "results": [
                        {
                            "seed": seed,
                            "boss_type": 2,
                            "boss_name": "DEATH_METAL",
                            "instance_id": f"worker-{index % 8:04d}",
                            "run_id": f"run-{seed}",
                            "session_id": "session",
                            "launch_id": f"launch-{index % 8}",
                            "curriculum_reset": reset,
                        }
                        for index, seed in enumerate(candidates)
                    ],
                }
            ),
            encoding="utf-8",
        )
        reports[bank] = path, candidates
    training = root / "training"
    training.mkdir()
    (training / "seed-selection.json").write_text(
        json.dumps(
            {
                "seeds": TRAINING_SEEDS,
                "boss_type": 2,
                "disclosure": "boss identity only",
                "source_report_sha256": sha256_file(reports["training"][0]),
            }
        ),
        encoding="utf-8",
    )
    evaluation = root / "evaluation"
    evaluation.mkdir()
    (evaluation / "heldout-selection.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "boss_type": 2,
                "disclosure": "boss identity only",
                "source_report_sha256": sha256_file(reports["evaluation"][0]),
            }
        ),
        encoding="utf-8",
    )
    return evaluation


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
        "checkpoint_global_step": 122880,
        "checkpoint_training_seed_schedule": "uniform-pool-v1",
        "checkpoint_training_seed_pool": TRAINING_SEEDS,
        "checkpoint_curriculum": {
            "start_level": 4,
            "target_level": 5,
            "profile": "player20",
            "reset_semantics": "normal-reset-sequential-goto-reward-reset-v1",
        },
        "checkpoint_freeze_base_updates": 10,
        "checkpoint_initialization": {
            "sha256": SOURCE_SHA256,
            "architecture_upgrade": "v2_to_v8_actor_parity_fresh_critic",
        },
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
    evaluation = _write_selections(tmp_path, seeds)
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
    evaluation = _write_selections(tmp_path, seeds)
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
    evaluation = _write_selections(tmp_path, seeds)
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


def test_guide_gate_rejects_unbound_or_wrong_initializer(tmp_path: Path) -> None:
    seeds = list(range(81001, 81025))
    evaluation = _write_selections(tmp_path, seeds)
    for name in ("parent", *TRIALS):
        for mode in MODES:
            directory = evaluation / name / mode
            directory.mkdir(parents=True)
            report = _report(seeds, set(), damage=3, mode=mode)
            if name == TRIALS[0] and mode == "deterministic":
                report["checkpoint_initialization"]["sha256"] = "0" * 64
            (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")

    try:
        compare(tmp_path)
    except ValueError as error:
        assert "initializer checkpoint mismatch" in str(error)
    else:
        raise AssertionError("wrong initializer provenance was accepted")


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
    assert '"-m", "autodancer.training.boss_identity"' in source
    assert "$report.results" in source
    assert '"--max-steps", "1"' not in source
    assert "autodancer.training.baseline" in source
    assert "protocol_schema_version -ne 10" in source
    assert "does not preserve the declared candidate bank" in source
    assert "leaks gameplay outcome fields" in source
    assert "Select-DeathMetalSeeds $trainingCalibration 48 $guideTrainingCandidates" in source
