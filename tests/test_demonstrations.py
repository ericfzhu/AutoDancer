from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from autodancer.training.demonstrations import (
    build_demonstration_bank,
    iter_trace_actions,
    load_successful_episode_traces,
    load_successful_evaluation_traces,
    validate_demonstration_bank,
    write_demonstration_bank,
)


def episode(*, seed: int = 92043, actions: list[int] | None = None) -> dict[str, object]:
    sequence = actions or [0, 1, 4, 5]
    return {
        "schema_version": 1,
        "seed": seed,
        "status": "curriculum_complete",
        "infrastructure_valid": True,
        "run_id": f"{seed}:2:44",
        "worker_id": "worker-0002",
        "policy_version": 71,
        "global_step": 72704,
        "turns": len(sequence),
        "furthest_zone": 2,
        "furthest_floor": 1,
        "curriculum_reset": {
            "id": "full-boss",
            "start_level": 4,
            "target_level": 5,
            "profile": "player20",
        },
        "boss_progress": {"initial_health": 9, "minimum_health": 0, "reached": True},
        "event_counts": {"enemy_damage": 9, "enemy_kill": 1},
        "successful_action_sequence": sequence,
    }


def write_journal(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_build_validate_and_atomically_write_bank(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    failed = {**episode(seed=1), "status": "dead", "successful_action_sequence": None}
    write_journal(first, [failed, episode(), episode()])
    write_journal(second, [episode(seed=92044, actions=[3, 2, 5])])

    bank = build_demonstration_bank([first, second])

    assert [trace["seed"] for trace in bank["traces"]] == [92043, 92044]
    assert bank["sources"][0]["successful_trace_count"] == 2
    validate_demonstration_bank(bank)
    destination = tmp_path / "bank.json"
    write_demonstration_bank(destination, bank)
    assert json.loads(destination.read_text(encoding="utf-8")) == bank


def test_successful_trace_rejects_invalid_or_incomplete_evidence(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    invalid = episode(actions=[0, 11])
    write_journal(path, [invalid])
    with pytest.raises(ValueError, match="invalid action"):
        load_successful_episode_traces(path)

    invalid = episode()
    invalid["turns"] = 99
    write_journal(path, [invalid])
    with pytest.raises(ValueError, match="turn count"):
        load_successful_episode_traces(path)

    invalid = episode()
    invalid["infrastructure_valid"] = False
    write_journal(path, [invalid])
    with pytest.raises(ValueError, match="infrastructure-valid"):
        load_successful_episode_traces(path)


def test_prefixed_success_is_not_misrepresented_as_a_full_reset_trace(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    prefixed = episode()
    prefixed["natural_prefix"] = {"kind": "qualified-live-trace-prefix-v1"}
    write_journal(path, [prefixed])
    assert load_successful_episode_traces(path) == ()


def test_extracts_full_reset_success_from_evaluation_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "controller_valid": True,
                "action_contract": "current",
                "checkpoint_updates": 90,
                "checkpoint_global_step": 92160,
                "curriculum_start_level": 4,
                "curriculum_target_level": 5,
                "curriculum_profile": "player20",
                "trained": {
                    "results": [
                        {
                            "seed": 92043,
                            "worker_id": "worker-0002",
                            "run_id": "92043:1:4",
                            "status": "curriculum_complete",
                            "turns": 4,
                            "furthest_zone": 2,
                            "furthest_floor": 1,
                            "boss_type": 2,
                            "initial_boss_health": 9,
                            "minimum_boss_health": 0,
                            "boss_damage": 9,
                            "boss_actor_types": [1, 2, 3, 4],
                            "death_metal_phase4_reached": True,
                            "event_counts": {"enemy_damage": 3, "enemy_kill": 1},
                            "successful_action_sequence": [0, 1, 4, 5],
                            "natural_prefix": {},
                        },
                        {"seed": 92044, "status": "dead"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    traces = load_successful_evaluation_traces(path)
    assert len(traces) == 1
    assert traces[0].seed == 92043
    assert traces[0].source_policy_version == 90
    assert traces[0].curriculum_reset["id"] == "fixed"
    assert traces[0].curriculum_reset["profile"] == "player20"


def test_bank_hash_and_trace_identity_are_tamper_evident(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    write_journal(path, [episode()])
    bank = build_demonstration_bank([path])

    tampered = copy.deepcopy(bank)
    tampered["traces"][0]["action_sequence"][0] = 2
    with pytest.raises(ValueError, match="bank hash mismatch"):
        validate_demonstration_bank(tampered)

    tampered = copy.deepcopy(bank)
    tampered["traces"][0]["trace_id"] = "0" * 64
    import hashlib

    unhashed = {key: value for key, value in tampered.items() if key != "bank_sha256"}
    tampered["bank_sha256"] = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="trace identity mismatch"):
        validate_demonstration_bank(tampered)


def test_iter_trace_actions_validates_at_consumption() -> None:
    assert list(iter_trace_actions({"action_sequence": [0, 5, 10]})) == [0, 5, 10]
    with pytest.raises(ValueError, match="invalid action"):
        list(iter_trace_actions({"action_sequence": [0, -1]}))
