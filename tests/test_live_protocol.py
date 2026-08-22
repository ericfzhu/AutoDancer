from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autodancer.constants import Action, GridChannel, PlayerFeature, Terrain
from autodancer.envs.live import AutoDancerLiveEnv
from autodancer.live.bridge import BridgeCommand
from autodancer.live.protocol import (
    LOG_MARKER,
    JsonlTurnSource,
    NativePipeTurnSource,
    ProtocolError,
    QueueTurnSource,
    validate_record,
)


class FakeReceiver:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = payloads

    def receive(self, timeout: float = 10.0, *, max_bytes: int = 65536) -> bytes:
        del timeout
        payload = self.payloads.pop(0)
        assert len(payload) <= max_bytes
        return payload


class FakeBridge:
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.restarts = 0

    def send_action(self, action: Action) -> BridgeCommand:
        self.actions.append(action)
        return BridgeCommand("test-session", len(self.actions), action, "ACTION", "worker-0000")

    def reset(self, seed: int) -> BridgeCommand:
        self.restarts += 1
        return BridgeCommand("test-session", self.restarts, None, "RESET", "worker-0000", seed)

    def restart(self) -> BridgeCommand:
        return self.reset(7)


def record(
    sequence: int,
    kind: str,
    *,
    status: str = "running",
    run_id: str = "run-7",
    events: list[dict] | None = None,
    requested_action: Action | None = None,
    command_id: int | None = None,
) -> dict:
    terminal = status in {"won", "dead"}
    truncated = status == "aborted"
    mask = np.zeros(11, dtype=int)
    if status == "running":
        mask[:4] = 1
    player = np.zeros(16, dtype=int)
    player[PlayerFeature.HEALTH] = 6
    player[PlayerFeature.MAX_HEALTH] = 6
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 1
    player[PlayerFeature.WON] = int(status == "won")
    player[PlayerFeature.DEAD] = int(status == "dead")
    return {
        "schema_version": 5,
        "instance_id": "worker-0000",
        "role": "worker",
        "run_id": run_id,
        "sequence": sequence,
        "kind": kind,
        "game": {"version": "v4.2.1-b5713", "steam_build": "22938426"},
        "character": "Bard",
        "seed": 7,
        "zone": 1,
        "floor": 1,
        "observation": {
            "grid": np.zeros((21, 21, 11), dtype=int).tolist(),
            "player": player.tolist(),
            "inventory": np.zeros((8, 4), dtype=int).tolist(),
            "action_mask": mask.tolist(),
        },
        "events": events or [],
        "episode_status": status,
        "terminated": terminal,
        "truncated": truncated,
        "metrics": {"turns": sequence},
        "bridge": {
            "kind": "RESET",
            "session_id": "test-session",
            "command_id": 1,
            "seed": 7,
        }
        if kind == "reset"
        else None
        if requested_action is None
        else {
            "kind": "ACTION",
            "session_id": "test-session",
            "command_id": sequence if command_id is None else command_id,
            "requested_action": int(requested_action),
            "engine_action": 1,
            "observed_action": 1,
        },
    }


def test_native_pipe_source_accepts_large_schema5_records_without_log_markers() -> None:
    reset = record(0, "reset")
    turn = record(1, "turn", requested_action=Action.UP)
    encoded = [json.dumps(item).encode("utf-8") for item in (reset, turn)]
    assert all(len(item) > 4096 for item in encoded)
    source = NativePipeTurnSource(FakeReceiver(encoded))
    assert source.read(1)["kind"] == "reset"
    assert source.read(1)["sequence"] == 1


def test_live_victory_terminates_and_reports_completion() -> None:
    sender = FakeBridge()
    source = QueueTurnSource(
        [
            record(0, "reset"),
            record(
                1,
                "terminal",
                status="won",
                events=[{"kind": "success", "amount": 1, "data": {"task_complete": True}}],
                requested_action=Action.RIGHT,
            ),
        ]
    )
    environment = AutoDancerLiveEnv(
        turn_source=source,
        bridge=sender,
        max_turns=10,
    )
    environment.reset(seed=7)
    observation, reward, terminated, truncated, info = environment.step(Action.RIGHT)
    assert terminated and not truncated
    assert info["episode_status"] == "won"
    assert info["completed"] == 1
    assert observation["player"][PlayerFeature.WON] == 1
    assert reward == pytest.approx(49.985)
    assert info["reward_components"]["victory"] == 50.0
    with pytest.raises(RuntimeError, match="episode ended"):
        environment.step(Action.RIGHT)


def test_live_adapter_rederives_deployment_features() -> None:
    payload = record(0, "reset")
    grid = np.asarray(payload["observation"]["grid"], dtype=int)
    grid[10, 10, GridChannel.TERRAIN_CLASS] = Terrain.STAIRS
    grid[10, 11, GridChannel.ACTOR_CLASS] = 2
    grid[10, 11, GridChannel.VISIBILITY] = 2
    payload["observation"]["grid"] = grid.tolist()
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource([payload]),
        bridge=FakeBridge(),
        attach_existing=True,
    )
    observation, _ = environment.reset()
    assert observation["player"][PlayerFeature.VISIBLE_ENEMIES] == 1
    assert observation["player"][PlayerFeature.ON_STAIRS] == 1
    assert observation["player"][PlayerFeature.TASK] == 0


def test_live_environment_applies_client_turn_limit() -> None:
    source = QueueTurnSource([record(0, "reset"), record(1, "turn", requested_action=Action.UP)])
    environment = AutoDancerLiveEnv(
        turn_source=source,
        bridge=FakeBridge(),
        max_turns=1,
    )
    environment.reset(seed=7)
    _, reward, terminated, truncated, info = environment.step(Action.UP)
    assert not terminated and truncated
    assert info["episode_status"] == "aborted"
    assert info["client_turn_limit"] == 1
    assert reward == pytest.approx(-1.015)
    assert info["reward_components"]["aborted"] == -1.0


def test_terminal_record_may_mask_every_action() -> None:
    validate_record(record(3, "terminal", status="won"))
    payload = record(0, "reset")
    payload["observation"]["action_mask"] = [0] * 11
    with pytest.raises(ProtocolError, match="cannot mask every action"):
        validate_record(payload)


def test_protocol_detects_lost_turn_and_run_change() -> None:
    source = QueueTurnSource([record(0, "reset"), record(2, "turn")])
    source.read()
    with pytest.raises(ProtocolError, match="expected 1"):
        source.read()

    source = QueueTurnSource([record(0, "reset"), record(1, "turn", run_id="another-run")])
    source.read()
    with pytest.raises(ProtocolError, match="Run identity changed"):
        source.read()


def test_protocol_rejects_invalid_observation_and_event() -> None:
    payload = record(0, "reset")
    payload["observation"]["action_mask"][0] = 2
    with pytest.raises(ProtocolError, match="only 0 or 1"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["observation"]["inventory"][0][1] = 4096
    with pytest.raises(ProtocolError, match="exact type identifier"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["events"] = [{"kind": "enemy_damage", "amount": -1}]
    with pytest.raises(ProtocolError, match="at least 0"):
        validate_record(payload)


def test_log_source_establishes_boundary_before_fast_restart(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    path.write_text("old line\n", encoding="utf-8")
    source = JsonlTurnSource(path)
    source.reset_sequence()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(LOG_MARKER + json.dumps(record(0, "reset")) + "\n")
    assert source.read(timeout=0.2)["sequence"] == 0


def test_live_environment_uses_shared_schema_and_reward_mapping() -> None:
    sender = FakeBridge()
    source = QueueTurnSource(
        [
            record(0, "reset"),
            record(
                1,
                "turn",
                events=[{"kind": "enemy_damage", "amount": 1}],
                requested_action=Action.RIGHT,
            ),
        ]
    )
    environment = AutoDancerLiveEnv(turn_source=source, bridge=sender)
    observation, info = environment.reset(seed=7)
    assert environment.observation_space.contains(observation)
    assert sender.restarts == 1
    assert info["protocol_schema_version"] == 5
    _, reward, terminated, truncated, info = environment.step(Action.RIGHT)
    assert sender.actions == [Action.RIGHT]
    assert reward == pytest.approx(0.01)
    assert info["reward_components"]["enemy_damage"] == 0.025
    assert not terminated and not truncated
    assert info["episode_status"] == "running"


def test_live_reset_ignores_post_death_turn_before_reset() -> None:
    source = QueueTurnSource(
        [
            record(83, "turn", run_id="dead-run"),
            record(0, "reset", run_id="restarted-run"),
        ]
    )
    environment = AutoDancerLiveEnv(turn_source=source, bridge=FakeBridge())
    _, info = environment.reset(seed=7)
    assert info["run_id"] == "restarted-run"


def test_queue_source_accepts_a_fresh_reset_after_attach() -> None:
    source = QueueTurnSource([record(7, "turn"), record(0, "reset", run_id="new-run")])
    source.read_latest()
    assert source.read()["sequence"] == 0


def test_log_source_reads_marker_and_logger_suffix(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    path.write_text(
        "unrelated log\n[Debug] '" + LOG_MARKER + json.dumps(record(0, "reset")) + "'\n",
        encoding="utf-8",
    )
    source = JsonlTurnSource(path, start_at_end=False)
    assert source.read(timeout=0.2)["run_id"] == "run-7"


def test_log_source_waits_for_a_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    payload = LOG_MARKER + json.dumps(record(0, "reset"))
    path.write_text(payload[:-10], encoding="utf-8")
    source = JsonlTurnSource(path, start_at_end=False)
    with pytest.raises(TimeoutError):
        source.read(timeout=0.02)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload[-10:] + "\n")
    assert source.read(timeout=0.2)["sequence"] == 0


def test_log_source_can_attach_to_latest_record(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    path.write_text(
        "\n".join(
            [
                LOG_MARKER + json.dumps(record(3, "turn")),
                LOG_MARKER + json.dumps(record(4, "turn")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = JsonlTurnSource(path)
    assert source.read_latest(timeout=0.2)["sequence"] == 4
    with path.open("a", encoding="utf-8") as handle:
        handle.write(LOG_MARKER + json.dumps(record(5, "turn")) + "\n")
    assert source.read(timeout=0.2)["sequence"] == 5


def test_live_environment_can_attach_without_restart() -> None:
    sender = FakeBridge()
    source = QueueTurnSource(
        [
            record(7, "turn"),
            record(8, "turn", requested_action=Action.LEFT, command_id=1),
        ]
    )
    environment = AutoDancerLiveEnv(
        turn_source=source,
        bridge=sender,
        attach_existing=True,
    )
    _, info = environment.reset()
    assert info["sequence"] == 7
    assert sender.restarts == 0
    environment.step(Action.LEFT)
    assert sender.actions == [Action.LEFT]


def test_placeholder_build_is_rejected() -> None:
    payload = record(0, "reset")
    payload["game"]["version"] = "SET_GAME_VERSION"
    with pytest.raises(ProtocolError, match="Unsupported game build"):
        validate_record(payload)


def test_schema_five_rejects_cross_worker_and_mismatched_seed() -> None:
    payload = record(0, "reset")
    payload["instance_id"] = "worker-0001"
    environment = AutoDancerLiveEnv(turn_source=QueueTurnSource([payload]), bridge=FakeBridge())
    with pytest.raises(ProtocolError, match="Expected worker 'worker-0000'"):
        environment.reset(seed=7)

    payload = record(0, "reset")
    payload["seed"] = 8
    environment = AutoDancerLiveEnv(turn_source=QueueTurnSource([payload]), bridge=FakeBridge())
    with pytest.raises(ProtocolError, match="acknowledgement mismatch"):
        environment.reset(seed=7)
