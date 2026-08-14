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
_JSON_DECODER = json.JSONDecoder()


class ProtocolError(RuntimeError):
    pass


def _decode_record_line(line: str, marker_index: int) -> dict[str, Any]:
    """Decode one record while ignoring the game logger's trailing quote."""
    payload = line[marker_index + len(LOG_MARKER) :].lstrip()
    record, _ = _JSON_DECODER.raw_decode(payload)
    if not isinstance(record, dict):
        raise ProtocolError("An AutoDancer log record must be a JSON object")
    return record


class TurnSource(Protocol):
    def reset_sequence(self) -> None: ...

    def read(self, timeout: float) -> dict[str, Any]: ...

    def read_latest(self, timeout: float) -> dict[str, Any]: ...


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
                    while True:
                        line_offset = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        if not line.endswith("\n"):
                            self._offset = line_offset
                            break
                        self._offset = handle.tell()
                        marker_index = line.find(LOG_MARKER)
                        if marker_index < 0:
                            continue
                        record = _decode_record_line(line, marker_index)
                        self._validate_sequence(record)
                        return record
            time.sleep(0.01)
        raise TimeoutError(f"No AutoDancer turn record arrived within {timeout:.1f} seconds")

    def read_latest(self, timeout: float = 5.0) -> dict[str, Any]:
        """Attach to the newest complete record already present in the log."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.path.exists():
                latest: dict[str, Any] | None = None
                with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                    while True:
                        line_offset = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        if not line.endswith("\n"):
                            self._offset = line_offset
                            break
                        marker_index = line.find(LOG_MARKER)
                        if marker_index >= 0:
                            latest = _decode_record_line(line, marker_index)
                        self._offset = handle.tell()
                if latest is not None:
                    self._validate_sequence(latest)
                    return latest
            time.sleep(0.01)
        raise TimeoutError(
            f"No existing AutoDancer turn record was found within {timeout:.1f} seconds"
        )

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

    def read_latest(self, timeout: float = 5.0) -> dict[str, Any]:
        """Return the next queued state as the current attached state in tests."""
        return self.read(timeout)


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
            "Unsupported protocol schema "
            f"{record.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    if record.get("kind") not in {"reset", "turn"}:
        raise ProtocolError("Record kind must be 'reset' or 'turn'")
    if record.get("character") != "Bard":
        raise ProtocolError("AutoDancer-Live-v0 requires Bard")
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
