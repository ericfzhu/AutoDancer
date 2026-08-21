from __future__ import annotations

from pathlib import Path

from autodancer.constants import Action
from autodancer.live.bridge import CoordinatorBridge, FileCommandBridge


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
