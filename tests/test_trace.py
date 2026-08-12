from __future__ import annotations

import json
from pathlib import Path

from autodancer.constants import Action
from autodancer.envs.sim import AutoDancerSimEnv
from autodancer.live.trace import compare_trace


def test_simulator_replays_conformance_trace(tmp_path: Path) -> None:
    actions = [Action.RIGHT, Action.DOWN, Action.WAIT]
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=31)
    records = [
        {
            "kind": "header",
            "schema_version": 1,
            "game_version": "4.2.0",
            "steam_build": "12345678",
            "seed": 31,
            "task": "navigation",
        }
    ]
    for sequence, action in enumerate(actions, start=1):
        environment.step(action)
        records.append(
            {
                "kind": "turn",
                "sequence": sequence,
                "action": int(action),
                "state": environment.snapshot(),
            }
        )
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    assert compare_trace(path) == []


def test_trace_reports_field_difference(tmp_path: Path) -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=4)
    environment.step(Action.WAIT)
    state = environment.snapshot()
    state["player"]["health"] = 99
    records = [
        {
            "kind": "header",
            "schema_version": 1,
            "game_version": "4.2.0",
            "steam_build": "12345678",
            "seed": 4,
            "task": "navigation",
        },
        {"kind": "turn", "sequence": 1, "action": int(Action.WAIT), "state": state},
    ]
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    assert any("player.health" in difference for difference in compare_trace(path))

