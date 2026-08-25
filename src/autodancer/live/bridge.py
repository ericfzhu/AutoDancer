"""Command-file transport for one AutoDancer SYNCHRONY instance."""

from __future__ import annotations

import os
import secrets
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
    instance_id: str
    seed: int | None = None
    target_level: int | None = None


class ActionBridge(Protocol):
    def send_action(self, action: Action) -> BridgeCommand: ...

    def reset(self, seed: int) -> BridgeCommand: ...

    def goto_level(self, level: int) -> BridgeCommand: ...

    def restart(self) -> BridgeCommand: ...


class CommandTransport(Protocol):
    def send(self, payload: bytes, timeout: float = 10.0) -> None: ...


class NativePipeCommandBridge:
    """Publish commands to one in-process Lua bridge over a named pipe."""

    def __init__(
        self,
        transport: CommandTransport,
        *,
        instance_id: str = "worker-0000",
        session_id: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.transport = transport
        self.instance_id = _safe_identifier(instance_id, "instance_id")
        self.session_id = _safe_identifier(session_id or uuid.uuid4().hex, "session_id")
        self.timeout = timeout
        self._next_command_id = 1

    def _publish(
        self,
        kind: str,
        action: Action | None,
        *,
        argument: int,
        seed: int | None = None,
        target_level: int | None = None,
    ) -> BridgeCommand:
        command = BridgeCommand(
            session_id=self.session_id,
            command_id=self._next_command_id,
            action=action,
            kind=kind,
            instance_id=self.instance_id,
            seed=seed,
            target_level=target_level,
        )
        self._next_command_id += 1
        payload = f"{kind} {command.session_id} {command.command_id} {argument}\n".encode("ascii")
        self.transport.send(payload, self.timeout)
        return command

    def send_action(self, action: Action) -> BridgeCommand:
        selected = Action(action)
        return self._publish("ACTION", selected, argument=int(selected))

    def reset(self, seed: int) -> BridgeCommand:
        seed = int(seed)
        if not 0 <= seed <= 2**31 - 1:
            raise ValueError("seed must be in 0..2147483647")
        return self._publish("RESET", None, argument=seed, seed=seed)

    def goto_level(self, level: int) -> BridgeCommand:
        level = int(level)
        if level < 1:
            raise ValueError("level must be positive")
        return self._publish("GOTO", None, argument=level, target_level=level)

    def restart(self) -> BridgeCommand:
        return self.reset(secrets.randbelow(2**31))

    def start(self) -> BridgeCommand:
        return self.restart()


class FileCommandBridge:
    """Publish commands without replacing the file mounted by the game."""

    def __init__(
        self,
        path: str | Path,
        *,
        instance_id: str = "worker-0000",
        session_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.instance_id = _safe_identifier(instance_id, "instance_id")
        self.session_id = session_id or uuid.uuid4().hex
        _safe_identifier(self.session_id, "session_id")
        self._next_command_id = 1

    def _publish(
        self,
        kind: str,
        action: Action | None,
        *,
        argument: int,
        seed: int | None = None,
        target_level: int | None = None,
    ) -> BridgeCommand:
        command = BridgeCommand(
            session_id=self.session_id,
            command_id=self._next_command_id,
            action=action,
            kind=kind,
            instance_id=self.instance_id,
            seed=seed,
            target_level=target_level,
        )
        self._next_command_id += 1
        payload = f"{kind} {command.session_id} {command.command_id} {argument}\n"
        _write_mounted_file(self.path, payload)
        return command

    def send_action(self, action: Action) -> BridgeCommand:
        selected = Action(action)
        return self._publish("ACTION", selected, argument=int(selected))

    def reset(self, seed: int) -> BridgeCommand:
        seed = int(seed)
        if not 0 <= seed <= 2**31 - 1:
            raise ValueError("seed must be in 0..2147483647")
        return self._publish("RESET", None, argument=seed, seed=seed)

    def goto_level(self, level: int) -> BridgeCommand:
        level = int(level)
        if level < 1:
            raise ValueError("level must be positive")
        return self._publish("GOTO", None, argument=level, target_level=level)

    def restart(self) -> BridgeCommand:
        return self.reset(secrets.randbelow(2**31))

    def start(self) -> BridgeCommand:
        return self.restart()


class CoordinatorBridge:
    """Publish lifecycle commands to the coordinator game process."""

    def __init__(self, path: str | Path, *, session_id: str | None = None) -> None:
        self.path = Path(path)
        self.session_id = _safe_identifier(session_id or uuid.uuid4().hex, "session_id")
        self._next_command_id = 1

    def _publish(self, kind: str, worker_id: str) -> BridgeCommand:
        worker_id = _safe_identifier(worker_id, "worker_id")
        command = BridgeCommand(
            session_id=self.session_id,
            command_id=self._next_command_id,
            action=None,
            kind=kind,
            instance_id="coordinator",
        )
        self._next_command_id += 1
        _write_mounted_file(
            self.path,
            f"{kind} {command.session_id} {command.command_id} {worker_id}\n",
        )
        return command

    def spawn(self, worker_id: str) -> BridgeCommand:
        return self._publish("SPAWN", worker_id)

    def close(self, worker_id: str) -> BridgeCommand:
        return self._publish("CLOSE", worker_id)


def _safe_identifier(value: str, label: str) -> str:
    if not value or not all(character.isalnum() or character in "-_" for character in value):
        raise ValueError(f"{label} may contain only letters, digits, '-' and '_'")
    return value


def _write_mounted_file(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".lua":
        escaped = payload.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        payload = f'return {{payload="{escaped}"}}\n'
    encoded = payload.encode("ascii")
    mode = "r+b" if path.exists() else "w+b"
    with path.open(mode) as command_file:
        command_file.seek(0)
        command_file.write(encoded)
        command_file.truncate()
        command_file.flush()
        os.fsync(command_file.fileno())
