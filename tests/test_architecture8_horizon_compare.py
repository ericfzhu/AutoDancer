from __future__ import annotations

import json
from pathlib import Path

from autodancer.training import architecture8_horizon_compare as comparison


def _summary(progress: float) -> dict[str, float | int]:
    return {
        "episodes": 10,
        "mean_progress": progress,
        "death_rate": 0.2,
        "step_limit_rate": 0.5,
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


def _complete_curves(root: Path, a8_progress: dict[int, float]) -> None:
    frozen = _summary(1.0)
    frozen["step_limit_rate"] = 0.7
    path = root / "curve-evaluation/a2-frozen/report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "trained": frozen,
                "worker_restarts": 0,
                "controller_valid": True,
                "infrastructure_events": [],
            }
        ),
        encoding="utf-8",
    )
    for step in comparison.CURVE_STEPS:
        _report(
            root / f"curve-evaluation/a2-continuation/step-{step:08d}/report.json",
            1.0,
        )
        _report(
            root / f"curve-evaluation/a8-continuation/step-{step:08d}/report.json",
            a8_progress[step],
        )
    (root / "representation-final.json").write_text(
        '{"checkpoints":[{"all_new_groups_material":true}]}', encoding="utf-8"
    )


def test_horizon_rejects_incomplete_evidence(tmp_path: Path) -> None:
    result = comparison.compare_horizon(tmp_path)
    assert result["decision"] == "invalid_or_incomplete_horizon_evidence"


def test_horizon_requires_repeated_final_advantage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        comparison, "_training_health", lambda root, arm: {"complete": True, "valid": True}
    )
    _complete_curves(
        tmp_path,
        {30_720: 1.0, 61_440: 1.0, 122_880: 1.1, 250_880: 1.2},
    )
    result = comparison.compare_horizon(tmp_path)
    assert result["passed"]
    assert result["decision"] == "a8_ready_for_multiseed_confirmation"


def test_final_only_advantage_is_inconclusive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        comparison, "_training_health", lambda root, arm: {"complete": True, "valid": True}
    )
    _complete_curves(
        tmp_path,
        {30_720: 1.0, 61_440: 1.0, 122_880: 1.0, 250_880: 1.2},
    )
    result = comparison.compare_horizon(tmp_path)
    assert not result["passed"]
    assert result["decision"] == "a8_final_only_advantage_requires_replication"
