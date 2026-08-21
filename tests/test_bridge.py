from __future__ import annotations

from pathlib import Path

from autodancer.constants import Action
from autodancer.live.bridge import FileCommandBridge


def test_file_bridge_publishes_monotonic_action_commands(tmp_path: Path) -> None:
    path = tmp_path / "bridge-command.txt"
    bridge = FileCommandBridge(path, session_id="session-1")

    first = bridge.send_action(Action.UP)
    assert first.command_id == 1
    assert path.read_text(encoding="ascii") == "ACTION session-1 1 0\n"

    second = bridge.send_action(Action.BOMB)
    assert second.command_id == 2
    assert path.read_text(encoding="ascii") == "ACTION session-1 2 5\n"


def test_file_bridge_publishes_restart(tmp_path: Path) -> None:
    path = tmp_path / "bridge-command.txt"
    command = FileCommandBridge(path, session_id="session-2").restart()
    assert command.action is None
    assert path.read_text(encoding="ascii") == "RESTART session-2 1 -1\n"


def test_file_bridge_publishes_all_zones_bard_start(tmp_path: Path) -> None:
    path = tmp_path / "bridge-command.txt"
    command = FileCommandBridge(path, session_id="session-3").start()
    assert command.kind == "START"
    assert path.read_text(encoding="ascii") == "START session-3 1 -1\n"
