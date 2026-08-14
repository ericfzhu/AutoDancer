from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from autodancer.constants import Action
from autodancer.envs.live import AutoDancerLiveEnv
from autodancer.live.protocol import (
    LOG_MARKER,
    JsonlTurnSource,
    ProtocolError,
    QueueTurnSource,
    validate_record,
)


class FakeSender:
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.restarts = 0

    def send_action(self, action: Action) -> None:
        self.actions.append(action)

    def restart(self) -> None:
        self.restarts += 1


class FakeCapture:
    def capture(self) -> np.ndarray:
        return np.zeros((256, 256, 3), dtype=np.uint8)


class RepeatingSource:
    def __init__(self) -> None:
        self.sequence = 0
        self.needs_reset = True

    def reset_sequence(self) -> None:
        self.sequence = 0
        self.needs_reset = True

    def read(self, timeout: float = 5.0) -> dict:
        del timeout
        payload = record(self.sequence, "reset" if self.needs_reset else "turn")
        self.needs_reset = False
        self.sequence += 1
        return payload

    def read_latest(self, timeout: float = 5.0) -> dict:
        return self.read(timeout)


def record(sequence: int, kind: str, events: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "kind": kind,
        "game": {"version": "4.2.0", "steam_build": "12345678"},
        "character": "Bard",
        "seed": 7,
        "zone": 1,
        "floor": 1,
        "observation": {
            "grid": np.zeros((21, 21, 7), dtype=int).tolist(),
            "player": np.zeros(16, dtype=int).tolist(),
            "inventory": np.zeros((8, 3), dtype=int).tolist(),
            "action_mask": np.ones(11, dtype=int).tolist(),
        },
        "events": events or [],
        "terminated": False,
        "truncated": False,
        "metrics": {"turns": sequence},
    }


def test_live_environment_uses_same_schema_and_action_mapping() -> None:
    sender = FakeSender()
    source = QueueTurnSource(
        [record(0, "reset"), record(1, "turn", [{"kind": "enemy_damage", "amount": 1}])]
    )
    environment = AutoDancerLiveEnv(
        turn_source=source,
        action_sender=sender,
        frame_capture=FakeCapture(),
        render_mode="rgb_array",
    )
    observation, info = environment.reset()
    assert environment.observation_space.contains(observation)
    assert sender.restarts == 1
    assert info["game"]["steam_build"] == "12345678"
    _, reward, terminated, truncated, info = environment.step(Action.RIGHT)
    assert sender.actions == [Action.RIGHT]
    assert reward == pytest.approx(0.049)
    assert not terminated and not truncated
    assert environment.render().shape == (256, 256, 3)


def test_live_environment_passes_gymnasium_check_with_fake_game() -> None:
    environment = AutoDancerLiveEnv(
        turn_source=RepeatingSource(), action_sender=FakeSender(), frame_capture=FakeCapture()
    )
    check_env(environment, skip_render_check=True)


def test_live_protocol_detects_lost_turn() -> None:
    source = QueueTurnSource([record(0, "reset"), record(2, "turn")])
    source.read()
    with pytest.raises(ProtocolError, match="expected 1"):
        source.read()


def test_log_source_reads_marker(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    path.write_text("unrelated log\n" + LOG_MARKER + json.dumps(record(0, "reset")) + "\n")
    source = JsonlTurnSource(path, start_at_end=False)
    assert source.read(timeout=0.2)["sequence"] == 0


def test_log_source_ignores_game_logger_suffix(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    path.write_text(
        "[Debug] [info] '" + LOG_MARKER + json.dumps(record(0, "reset")) + "'\n"
    )
    source = JsonlTurnSource(path, start_at_end=False)
    assert source.read(timeout=0.2)["sequence"] == 0


def test_log_source_waits_for_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    payload = LOG_MARKER + json.dumps(record(0, "reset"))
    path.write_text(payload[:-10])
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
        + "\n"
    )
    source = JsonlTurnSource(path)
    assert source.read_latest(timeout=0.2)["sequence"] == 4
    with path.open("a", encoding="utf-8") as handle:
        handle.write(LOG_MARKER + json.dumps(record(5, "turn")) + "\n")
    assert source.read(timeout=0.2)["sequence"] == 5


def test_live_environment_can_attach_without_restart() -> None:
    sender = FakeSender()
    source = QueueTurnSource([record(7, "turn"), record(8, "turn")])
    environment = AutoDancerLiveEnv(
        turn_source=source,
        action_sender=sender,
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
    with pytest.raises(ProtocolError, match="pin the game"):
        validate_record(payload)
