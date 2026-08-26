from __future__ import annotations

import json
from pathlib import Path

from autodancer.training import architecture8_replication_compare as comparison


def _summary(progress: float) -> dict[str, float | int]:
    return {
        "episodes": 10,
        "mean_progress": progress,
        "death_rate": 0.2,
        "step_limit_rate": 0.1,
        "unchanged_position_rate": 0.2,
        "enemy_kills": 20,
        "item_pickups": 10,
    }


def _report(path: Path, progress: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "trained": _summary(progress),
                "worker_restarts": 0,
                "controller_valid": True,
                "infrastructure_events": [],
            }
        ),
        encoding="utf-8",
    )


def test_replication_stops_when_evidence_is_missing(tmp_path: Path) -> None:
    result = comparison.compare_replication(tmp_path)
    assert result["decision"] == "stop_before_broad_gameplay"
    assert not result["curve_gate_passed"]


def test_replication_curve_and_broad_gates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        comparison,
        "_training_health",
        lambda root, arm: {"complete": True, "valid": True},
    )
    (tmp_path / "parity.json").write_text('{"passed": true}', encoding="utf-8")
    for name in ("warmup", "final"):
        (tmp_path / f"representation-{name}.json").write_text(
            '{"checkpoints":[{"all_new_groups_material":true}]}',
            encoding="utf-8",
        )
    _report(tmp_path / "curve-evaluation/a2-frozen/report.json", 1.0)
    for arm, progress in (("a2-finetune", 1.0), ("a8-candidate", 1.1)):
        for step in comparison.CURVE_STEPS[1:]:
            _report(
                tmp_path / f"curve-evaluation/{arm}/step-{step:08d}/report.json",
                progress,
            )

    curve = comparison.compare_replication(tmp_path)
    assert curve["curve_gate_passed"]
    assert curve["decision"] == "ready_for_broad_gameplay"
    assert [point["source_arm"] for point in curve["curves"]["a8-candidate"]] == [
        "a2-frozen",
        "a8-candidate",
        "a8-candidate",
        "a8-candidate",
    ]

    for arm, progress in (
        ("a2-frozen", 1.0),
        ("a2-finetune", 1.05),
        ("a8-candidate", 1.2),
    ):
        _report(tmp_path / f"broad-evaluation/{arm}/report.json", progress)
    broad = comparison.compare_replication(tmp_path)
    assert broad["broad_gate_passed"]
    assert broad["decision"] == "a8_ready_for_multiseed_confirmation"


def test_controller_invalidates_replication_evaluation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        comparison,
        "_training_health",
        lambda root, arm: {"complete": True, "valid": True},
    )
    (tmp_path / "parity.json").write_text('{"passed": true}', encoding="utf-8")
    for name in ("warmup", "final"):
        (tmp_path / f"representation-{name}.json").write_text(
            '{"checkpoints":[{"all_new_groups_material":true}]}',
            encoding="utf-8",
        )
    _report(tmp_path / "curve-evaluation/a2-frozen/report.json", 1.0)
    for arm in comparison.TRAINED_ARMS:
        for step in comparison.CURVE_STEPS[1:]:
            path = tmp_path / f"curve-evaluation/{arm}/step-{step:08d}/report.json"
            _report(path, 1.0)
    bad_path = tmp_path / "curve-evaluation/a8-candidate/step-00010240/report.json"
    bad = json.loads(bad_path.read_text())
    bad["controller_valid"] = False
    bad_path.write_text(json.dumps(bad), encoding="utf-8")

    result = comparison.compare_replication(tmp_path)
    assert not result["curve_criteria"]["valid_controller_evaluations"]
    assert not result["curve_gate_passed"]
