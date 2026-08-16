from __future__ import annotations

import json
from pathlib import Path

from autodancer.constants import Action
from autodancer.envs.sim import AutoDancerSimEnv
from autodancer.live.trace import TraceWriter, compare_trace, load_trace


def test_legacy_state_trace_still_replays(tmp_path: Path) -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=31)
    records = [
        {
            "kind": "header",
            "schema_version": 1,
            "game_version": "4.2.1",
            "steam_build": "22938426",
            "seed": 31,
            "task": "navigation",
        }
    ]
    for sequence, action in enumerate((Action.RIGHT, Action.DOWN), start=1):
        environment.step(action)
        records.append(
            {
                "kind": "turn",
                "sequence": sequence,
                "action": int(action),
                "state": environment.snapshot(),
            }
        )
    path = tmp_path / "legacy.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    assert compare_trace(path) == []


def test_trace_writer_preserves_live_evidence_without_asserting_it_by_default(
    tmp_path: Path,
) -> None:
    environment = AutoDancerSimEnv(task="navigation")
    observation, _ = environment.reset(seed=5)
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(
        path,
        info={
            "protocol_schema_version": 2,
            "game": {"version": "4.2.1", "steam_build": "22938426"},
            "run_id": "run-5",
            "seed": 5,
            "zone": 1,
            "floor": 1,
        },
        task="navigation",
        initial_observation=observation,
    )
    observation, reward, terminated, truncated, info = environment.step(Action.RIGHT)
    info = dict(info, sequence=1, episode_status="running")
    writer.append(
        action=Action.RIGHT,
        observation=observation,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )
    header, turns = load_trace(path)
    assert "initial_live_observation" in header
    assert "initial_observation" not in header
    assert "live_observation" in turns[0]
    assert "observation" not in turns[0]
    assert compare_trace(path) == []
    try:
        TraceWriter(path, info={}, task="navigation")
    except FileExistsError:
        pass
    else:
        raise AssertionError("TraceWriter unexpectedly overwrote an existing trace")


def test_schema_two_partial_state_reports_difference(tmp_path: Path) -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=4)
    environment.step(Action.WAIT)
    records = [
        {
            "kind": "header",
            "schema_version": 2,
            "protocol_schema_version": 2,
            "game_version": "4.2.1",
            "steam_build": "22938426",
            "run_id": "run-4",
            "seed": 4,
            "task": "navigation",
        },
        {
            "kind": "turn",
            "sequence": 1,
            "action": int(Action.WAIT),
            "state": {"player": {"health": 99}},
        },
    ]
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    assert any("player.health" in difference for difference in compare_trace(path))
