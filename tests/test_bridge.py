from __future__ import annotations

from pathlib import Path

import pytest

from autodancer.constants import Action
from autodancer.live.bridge import (
    CoordinatorBridge,
    FileCommandBridge,
    NativePipeCommandBridge,
)


class RecordingTransport:
    def __init__(self) -> None:
        self.messages: list[tuple[bytes, float]] = []

    def send(self, payload: bytes, timeout: float = 10.0) -> None:
        self.messages.append((payload, timeout))


def test_native_pipe_bridge_routes_commands_without_files() -> None:
    transport = RecordingTransport()
    bridge = NativePipeCommandBridge(
        transport,
        instance_id="worker-0007",
        session_id="session-native",
        timeout=2.5,
    )

    action = bridge.send_action(Action.LEFT)
    reset = bridge.reset(11001)
    goto = bridge.goto_level(4)

    assert action.instance_id == "worker-0007"
    assert reset.seed == 11001
    assert goto.target_level == 4
    assert transport.messages == [
        (b"ACTION session-native 1 3\n", 2.5),
        (b"RESET session-native 2 11001\n", 2.5),
        (b"GOTO session-native 3 4\n", 2.5),
    ]


def test_native_pipe_bridge_routes_assisted_curriculum_profile() -> None:
    transport = RecordingTransport()
    command = NativePipeCommandBridge(
        transport,
        instance_id="worker-0001",
        session_id="session-profile",
    ).goto_level(4, "boss1hp-player20")
    assert command.target_level == 4
    assert command.curriculum_profile == "boss1hp-player20"
    assert transport.messages == [(b"GOTO session-profile 1 4_boss1hp-player20\n", 10.0)]


def test_bridge_rejects_unknown_curriculum_profile(tmp_path: Path) -> None:
    bridge = FileCommandBridge(tmp_path / "bridge.txt", session_id="session-profile")
    with pytest.raises(ValueError, match="curriculum_profile"):
        bridge.goto_level(4, "make-boss-disappear")


def test_file_bridge_publishes_monotonic_action_commands(tmp_path: Path) -> None:
    path = tmp_path / "bridge-command.txt"
    bridge = FileCommandBridge(path, session_id="session-1")

    first = bridge.send_action(Action.UP)
    assert first.command_id == 1
    assert path.read_text(encoding="ascii") == "ACTION session-1 1 0\n"

    second = bridge.send_action(Action.BOMB)
    assert second.command_id == 2
    assert path.read_text(encoding="ascii") == "ACTION session-1 2 5\n"


def test_file_bridge_publishes_seeded_reset(tmp_path: Path) -> None:
    path = tmp_path / "bridge-command.txt"
    command = FileCommandBridge(path, session_id="session-2").reset(12345)
    assert command.action is None
    assert command.seed == 12345
    assert path.read_text(encoding="ascii") == "RESET session-2 1 12345\n"


def test_file_bridge_publishes_qualification_level_transition(tmp_path: Path) -> None:
    path = tmp_path / "bridge-command.txt"
    command = FileCommandBridge(path, session_id="session-2").goto_level(5)
    assert command.target_level == 5
    assert path.read_text(encoding="ascii") == "GOTO session-2 1 5\n"


def test_file_bridge_publishes_all_zones_bard_start(tmp_path: Path) -> None:
    path = tmp_path / "bridge-command.txt"
    command = FileCommandBridge(path, session_id="session-3").start()
    assert command.kind == "RESET"
    assert path.read_text(encoding="ascii").startswith("RESET session-3 1 ")


def test_coordinator_routes_worker_lifecycle_commands(tmp_path: Path) -> None:
    path = tmp_path / "bridge-command.coordinator.txt"
    bridge = CoordinatorBridge(path, session_id="supervisor")
    bridge.spawn("worker-0003")
    assert path.read_text(encoding="ascii") == "SPAWN supervisor 1 worker-0003\n"
    bridge.close("worker-0003")
    assert path.read_text(encoding="ascii") == "CLOSE supervisor 2 worker-0003\n"
