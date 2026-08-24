from __future__ import annotations

import json
from pathlib import Path

from autodancer.training import architecture8_compare


def _summary(*, progress: float, death: float = 0.2) -> dict[str, float | int]:
    return {
        "episodes": 10,
        "mean_progress": progress,
        "death_rate": death,
        "step_limit_rate": 0.1,
        "unchanged_position_rate": 0.2,
        "enemy_kills": 20,
        "item_pickups": 10,
    }


def _write_report(path: Path, summary: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"trained": summary, "worker_restarts": 0}), encoding="utf-8"
    )


def test_comparison_stops_cleanly_when_artifacts_are_missing(tmp_path: Path) -> None:
    report = architecture8_compare.compare_experiment(tmp_path)

    assert report["decision"] == "stop_before_broad_gameplay"
    assert not report["curve_gate_passed"]
    assert not report["broad_gate_passed"]


def test_comparison_requires_curve_gate_before_broad_gameplay(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        architecture8_compare,
        "_training_health",
        lambda root, arm: {"complete": True, "valid": True, "worker_restarts": 0},
    )
    for arm in architecture8_compare.ARMS:
        for step in architecture8_compare.CURVE_STEPS:
            source = "a2-fixed" if arm == "a8" and step == 0 else arm
            _write_report(
                tmp_path
                / "curve-evaluation"
                / f"{source}-step-{step:08d}.json",
                _summary(progress=1.0 if arm != "a8" else 1.1),
            )
    for name in ("warmup", "final"):
        (tmp_path / f"representation-{name}.json").write_text(
            json.dumps({"checkpoints": [{"all_new_groups_material": True}]}),
            encoding="utf-8",
        )

    curve_report = architecture8_compare.compare_experiment(tmp_path)
    assert curve_report["curve_gate_passed"]
    assert curve_report["decision"] == "ready_for_broad_gameplay"

    for arm, progress in (("a2-legacy", 1.0), ("a2-fixed", 1.05), ("a8", 1.2)):
        _write_report(
            tmp_path / "broad-evaluation" / f"{arm}.json",
            _summary(progress=progress),
        )
    broad_report = architecture8_compare.compare_experiment(tmp_path)
    assert broad_report["broad_gate_passed"]
    assert broad_report["decision"] == "a8_ready_for_multiseed_confirmation"


def test_a8_curve_reuses_fixed_control_only_at_step_zero(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        architecture8_compare,
        "_training_health",
        lambda root, arm: {"complete": True, "valid": True},
    )
    for step in architecture8_compare.CURVE_STEPS:
        _write_report(
            tmp_path / "curve-evaluation" / f"a2-fixed-step-{step:08d}.json",
            _summary(progress=1.0),
        )
        if step:
            _write_report(
                tmp_path / "curve-evaluation" / f"a8-step-{step:08d}.json",
                _summary(progress=1.0 + step / 100_000),
            )

    curves = architecture8_compare._curve(tmp_path)

    assert [point["source_arm"] for point in curves["a8"]] == [
        "a2-fixed",
        "a8",
        "a8",
        "a8",
    ]
    assert curves["a8"][0]["summary"]["mean_progress"] == 1.0
    assert curves["a8"][-1]["summary"]["mean_progress"] == 1.3072
