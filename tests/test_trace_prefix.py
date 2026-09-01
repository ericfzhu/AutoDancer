from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autodancer.training.demonstrations import build_demonstration_bank, write_demonstration_bank
from autodancer.training.trace_prefix import (
    QualifiedTracePrefixBank,
    parse_trace_tail_window,
)


def episode(seed: int, actions: list[int], *, policy_version: int) -> dict[str, object]:
    return {
        "seed": seed,
        "status": "curriculum_complete",
        "infrastructure_valid": True,
        "run_id": f"{seed}:1:2",
        "worker_id": "worker-0000",
        "policy_version": policy_version,
        "global_step": policy_version * 1024,
        "turns": len(actions),
        "furthest_zone": 2,
        "furthest_floor": 1,
        "curriculum_reset": {
            "id": "full-boss",
            "start_level": 4,
            "target_level": 5,
            "profile": "player20",
        },
        "boss_progress": {},
        "event_counts": {},
        "successful_action_sequence": actions,
    }


def fixtures(tmp_path: Path) -> tuple[Path, Path, dict]:
    journal = tmp_path / "episodes.jsonl"
    (tmp_path / "config.json").write_text(
        json.dumps({"action_contract": "current"}), encoding="utf-8"
    )
    journal.write_text(
        json.dumps(episode(7, [0, 1, 2, 3], policy_version=2)) + "\n",
        encoding="utf-8",
    )
    bank = build_demonstration_bank([journal])
    bank_path = tmp_path / "bank.json"
    write_demonstration_bank(bank_path, bank)
    trace = bank["traces"][0]
    report = {
        "schema_version": 1,
        "kind": "qualified-live-action-traces-report-v1",
        "bank_sha256": bank["bank_sha256"],
        "valid": True,
        "worker_restarts": 0,
        "results": [
            {
                "trace_id": trace["trace_id"],
                "valid": True,
                "actual": {"qualified_action_sequence": [0, 1, 2, 3]},
                "turn_digests": [
                    hashlib.sha256(str(index).encode()).hexdigest() for index in range(5)
                ],
            }
        ],
    }
    report_path = tmp_path / "qualification.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return bank_path, report_path, trace


def test_loads_only_freshly_qualified_prefixes_and_binds_identity(tmp_path: Path) -> None:
    bank_path, report_path, trace = fixtures(tmp_path)
    prefix = QualifiedTracePrefixBank.load(bank_path, report_path, tail_actions=2)
    assert prefix.seeds == (7,)
    assert prefix.action_contract == "current"
    assert prefix.trace_for_seed(7).trace_id == trace["trace_id"]
    assert prefix.specification()["prefix_actions"] == {"7": 2}
    assert len(prefix.qualification_sha256) == 64


def test_balanced_trace_window_is_deterministic_and_identified(tmp_path: Path) -> None:
    bank_path, report_path, _ = fixtures(tmp_path)
    prefix = QualifiedTracePrefixBank.load(
        bank_path,
        report_path,
        tail_actions=3,
        tail_action_window=(1, 2, 3),
    )
    assert [prefix.tail_for_episode(0, episode) for episode in range(6)] == [
        1,
        2,
        3,
        1,
        2,
        3,
    ]
    assert prefix.tail_for_episode(1, 0) == 2
    specification = prefix.specification()
    assert specification["schema_version"] == 2
    assert specification["kind"] == "qualified-live-trace-window-v2"
    assert specification["tail_action_window"] == [1, 2, 3]
    assert specification["prefix_actions_by_tail"] == {"7": {"1": 3, "2": 2, "3": 1}}


def test_trace_window_parser_rejects_invalid_values() -> None:
    assert parse_trace_tail_window("3, 1,2") == (1, 2, 3)
    with pytest.raises(ValueError, match="duplicates"):
        parse_trace_tail_window("1,1")
    with pytest.raises(ValueError, match="positive"):
        parse_trace_tail_window("0,1")


def test_rejects_unqualified_or_empty_prefix(tmp_path: Path) -> None:
    bank_path, report_path, _ = fixtures(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["valid"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="not valid"):
        QualifiedTracePrefixBank.load(bank_path, report_path, tail_actions=2)

    _, report_path, _ = fixtures(tmp_path)
    with pytest.raises(ValueError, match="leaves no qualified prefix"):
        QualifiedTracePrefixBank.load(bank_path, report_path, tail_actions=4)


def test_loads_fresh_replay_canonical_prefix_and_rejects_nonprefix(tmp_path: Path) -> None:
    bank_path, report_path, _ = fixtures(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = report["results"][0]
    result["actual"]["qualified_action_sequence"] = [0, 1, 2]
    result["turn_digests"] = result["turn_digests"][:4]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    prefix = QualifiedTracePrefixBank.load(bank_path, report_path, tail_actions=1)
    assert prefix.trace_for_seed(7).actions == (0, 1, 2)
    assert prefix.specification()["prefix_actions"] == {"7": 2}

    result["actual"]["qualified_action_sequence"] = [0, 3, 2]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="not a source prefix"):
        QualifiedTracePrefixBank.load(bank_path, report_path, tail_actions=1)
