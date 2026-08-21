"""Command-file transport for one AutoDancer SYNCHRONY instance."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autodancer.constants import Action


@dataclass(frozen=True, slots=True)
class BridgeCommand:
    session_id: str
    command_id: int
    action: Action | None
    kind: str


class ActionBridge(Protocol):
    def send_action(self, action: Action) -> BridgeCommand: ...

    def restart(self) -> BridgeCommand: ...

    def start(self) -> BridgeCommand: ...


class FileCommandBridge:
    """Publish commands without replacing the file mounted by the game."""

    def __init__(self, path: str | Path, *, session_id: str | None = None) -> None:
        self.path = Path(path)
        self.session_id = session_id or uuid.uuid4().hex
        if not self.session_id or not all(c.isalnum() or c in "-_" for c in self.session_id):
            raise ValueError("session_id may contain only letters, digits, '-' and '_'")
        self._next_command_id = 1

    def _publish(self, kind: str, action: Action | None) -> BridgeCommand:
        command = BridgeCommand(
            session_id=self.session_id,
            command_id=self._next_command_id,
            action=action,
            kind=kind,
        )
        self._next_command_id += 1
        action_value = -1 if action is None else int(action)
        payload = f"{kind} {command.session_id} {command.command_id} {action_value}\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = payload.encode("ascii")
        mode = "r+b" if self.path.exists() else "w+b"
        with self.path.open(mode) as command_file:
            command_file.seek(0)
            command_file.write(encoded)
            command_file.truncate()
            command_file.flush()
            os.fsync(command_file.fileno())
        return command

    def send_action(self, action: Action) -> BridgeCommand:
        return self._publish("ACTION", Action(action))

    def restart(self) -> BridgeCommand:
        return self._publish("RESTART", None)

    def start(self) -> BridgeCommand:
        return self._publish("START", None)
