from __future__ import annotations

import json
from pathlib import Path

import pytest

from autodancer.training.recurrent_state_compare import compare_recurrent_states


def _report(mode: str, progress: list[tuple[int, int]], *, valid: bool = True) -> dict:
    seeds = [81_001, 81_002, 81_003]
    return {
        "schema_version": 2,
        "checkpoint_sha256": "abc123",
        "seeds": seeds,
        "policy_seed": 108_001,
        "policy_mode": "stochastic",
        "action_contract": "current",
        "recurrent_state_mode": mode,
        "curriculum_start_level": 1,
        "curriculum_target_level": None,
        "curriculum_profile": "normal",
        "max_steps_per_episode": 5000,
        "controller_valid": valid,
        "infrastructure_events": [],
        "trained": {
            "results": [
                {
                    "seed": seed,
                    "furthest_zone": zone,
                    "furthest_floor": floor,
                    "status": "dead" if index != 2 else "step_limit",
                    "turns": 100 + index,
                }
                for index, (seed, (zone, floor)) in enumerate(zip(seeds, progress, strict=True))
            ]
        },
    }


def _write(root: Path, mode: str, report: dict) -> None:
    directory = root / mode
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_recurrent_state_comparison_is_seed_paired(tmp_path: Path) -> None:
    _write(tmp_path, "carry", _report("carry", [(1, 1), (1, 2), (1, 3)]))
    _write(
        tmp_path,
        "reset-on-floor-transition",
        _report("reset-on-floor-transition", [(1, 1), (1, 3), (1, 2)]),
    )
    _write(
        tmp_path,
        "reset-every-step",
        _report("reset-every-step", [(1, 1), (1, 1), (2, 1)]),
    )

    result = compare_recurrent_states(tmp_path)

    assert result["valid"] is True
    assert result["summaries"]["carry"]["mean_progress"] == 2
    floor_reset = result["paired_against_carry"]["reset-on-floor-transition"]
    assert floor_reset["improved_seeds"] == [81_002]
    assert floor_reset["regressed_seeds"] == [81_003]
    assert floor_reset["carry_floor2_or_better_seeds"] == [81_002, 81_003]
    every_step = result["paired_against_carry"]["reset-every-step"]
    assert every_step["improved_seeds"] == [81_003]
    assert every_step["regressed_seeds"] == [81_002]
    assert result["summaries"]["reset-every-step"]["zone2_or_better"] == 1


def test_recurrent_state_comparison_rejects_mismatched_contract(tmp_path: Path) -> None:
    for mode in ("carry", "reset-on-floor-transition", "reset-every-step"):
        report = _report(mode, [(1, 1), (1, 2), (1, 3)])
        if mode == "reset-every-step":
            report["policy_seed"] = 108_002
        _write(tmp_path, mode, report)

    with pytest.raises(ValueError, match="policy_seed"):
        compare_recurrent_states(tmp_path)


def test_recurrent_state_comparison_rejects_controller_fault(tmp_path: Path) -> None:
    for mode in ("carry", "reset-on-floor-transition", "reset-every-step"):
        _write(
            tmp_path,
            mode,
            _report(mode, [(1, 1), (1, 2), (1, 3)], valid=mode != "carry"),
        )

    with pytest.raises(ValueError, match="Controller-invalid"):
        compare_recurrent_states(tmp_path)
