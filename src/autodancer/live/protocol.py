"""Line-based telemetry protocol for the SYNCHRONY debug log."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    PLAYER_FEATURES,
)

LOG_MARKER = "AUTODANCER_JSON:"
SCHEMA_VERSION = 1


class ProtocolError(RuntimeError):
    pass


class TurnSource(Protocol):
    def reset_sequence(self) -> None: ...

    def read(self, timeout: float) -> dict[str, Any]: ...


class JsonlTurnSource:
    """Tail records that the local Lua mod prints to the game debug log."""

    def __init__(self, path: str | Path, *, start_at_end: bool = True) -> None:
        self.path = Path(path)
        self.start_at_end = start_at_end
        self._offset: int | None = None
        self._expected_sequence: int | None = None

    def reset_sequence(self) -> None:
        self._expected_sequence = None

    def read(self, timeout: float = 5.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.path.exists():
                size = self.path.stat().st_size
                if self._offset is None:
                    self._offset = size if self.start_at_end else 0
                if size < self._offset:
                    self._offset = 0
                with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(self._offset)
                    for line in handle:
                        self._offset = handle.tell()
                        marker_index = line.find(LOG_MARKER)
                        if marker_index < 0:
                            continue
                        record = json.loads(line[marker_index + len(LOG_MARKER) :])
                        self._validate_sequence(record)
                        return record
            time.sleep(0.01)
        raise TimeoutError(f"No AutoDancer turn record arrived within {timeout:.1f} seconds")

    def _validate_sequence(self, record: dict[str, Any]) -> None:
        sequence = int(record.get("sequence", -1))
        if sequence < 0:
            raise ProtocolError("A turn record has no valid sequence number")
        if self._expected_sequence is not None and sequence != self._expected_sequence:
            raise ProtocolError(
                f"Turn sequence mismatch: expected {self._expected_sequence}, received {sequence}"
            )
        self._expected_sequence = sequence + 1


class QueueTurnSource:
    """In-memory source for tests and prerecorded live sessions."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = deque(records)
        self.expected_sequence: int | None = None

    def reset_sequence(self) -> None:
        self.expected_sequence = None

    def read(self, timeout: float = 5.0) -> dict[str, Any]:
        del timeout
        if not self.records:
            raise TimeoutError("The in-memory turn source is empty")
        record = self.records.popleft()
        sequence = int(record.get("sequence", -1))
        if self.expected_sequence is not None and sequence != self.expected_sequence:
            raise ProtocolError(
                f"Turn sequence mismatch: expected {self.expected_sequence}, received {sequence}"
            )
        self.expected_sequence = sequence + 1
        return record


def decode_observation(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    observation = {
        "grid": np.asarray(payload.get("grid"), dtype=np.int16),
        "player": np.asarray(payload.get("player"), dtype=np.int32),
        "inventory": np.asarray(payload.get("inventory"), dtype=np.int16),
        "action_mask": np.asarray(payload.get("action_mask"), dtype=np.int8),
    }
    expected = {
        "grid": (GRID_SIZE, GRID_SIZE, GRID_CHANNELS),
        "player": (PLAYER_FEATURES,),
        "inventory": (INVENTORY_SLOTS, INVENTORY_FEATURES),
        "action_mask": (ACTION_COUNT,),
    }
    for name, shape in expected.items():
        if observation[name].shape != shape:
            raise ProtocolError(
                f"Observation field {name!r} has shape {observation[name].shape}; expected {shape}"
            )
    return observation


def validate_record(record: dict[str, Any]) -> None:
    if int(record.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ProtocolError(
            f"Unsupported protocol schema {record.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    if record.get("kind") not in {"reset", "turn"}:
        raise ProtocolError("Record kind must be 'reset' or 'turn'")
    game = record.get("game", {})
    version = str(game.get("version", ""))
    steam_build = str(game.get("steam_build", ""))
    if (
        not version
        or not steam_build
        or version.startswith("SET_")
        or steam_build.startswith("SET_")
    ):
        raise ProtocolError("Each record must pin the game version and Steam build")
    decode_observation(record.get("observation", {}))
