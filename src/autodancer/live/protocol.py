"""Validated telemetry sources for the SYNCHRONY live bridge."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    MAP_SIZE,
    PLAYER_FEATURES,
    ActorKind,
    GridChannel,
    ItemKind,
    ObjectKind,
    PlayerFeature,
    StatusFlag,
    Terrain,
    TrapKind,
)

LOG_MARKER = "AUTODANCER_JSON:"
SCHEMA_VERSION = 9
SUPPORTED_GAME_VERSION = "v4.2.1-b5713"
SUPPORTED_STEAM_BUILD = "22938426"
EPISODE_STATUSES = frozenset({"running", "won", "dead", "aborted"})
RECORD_KINDS = frozenset({"reset", "turn", "terminal"})
_JSON_DECODER = json.JSONDecoder()


class ProtocolError(RuntimeError):
    pass


def _decode_record_line(line: str, marker_index: int) -> dict[str, Any]:
    payload = line[marker_index + len(LOG_MARKER) :].lstrip()
    try:
        record, _ = _JSON_DECODER.raw_decode(payload)
    except json.JSONDecodeError as error:
        raise ProtocolError("An AutoDancer log record contains invalid JSON") from error
    if not isinstance(record, dict):
        raise ProtocolError("An AutoDancer log record must be a JSON object")
    return record


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"{label} must be an integer, not a boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"{label} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise ProtocolError(f"{label} must be an integer")
    if minimum is not None and result < minimum:
        raise ProtocolError(f"{label} must be at least {minimum}")
    return result


def _integer_array(
    value: Any,
    label: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"Observation field {label!r} is not a rectangular array") from error
    if raw.shape != shape:
        raise ProtocolError(f"Observation field {label!r} has shape {raw.shape}; expected {shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ProtocolError(f"Observation field {label!r} must contain integers")
    return raw.astype(dtype, copy=False)


def decode_observation(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(payload, Mapping):
        raise ProtocolError("The observation must be an object")
    observation = {
        "grid": _integer_array(
            payload.get("grid"),
            "grid",
            (GRID_SIZE, GRID_SIZE, GRID_CHANNELS),
            np.dtype(np.int16),
        ),
        "player": _integer_array(
            payload.get("player"),
            "player",
            (PLAYER_FEATURES,),
            np.dtype(np.int32),
        ),
        "inventory": _integer_array(
            payload.get("inventory"),
            "inventory",
            (INVENTORY_SLOTS, INVENTORY_FEATURES),
            np.dtype(np.int16),
        ),
        "action_mask": _integer_array(
            payload.get("action_mask"),
            "action_mask",
            (ACTION_COUNT,),
            np.dtype(np.int8),
        ),
    }

    grid = observation["grid"]
    channel_ranges = {
        GridChannel.TERRAIN_CLASS: (0, len(Terrain) - 1),
        GridChannel.TERRAIN_TYPE: (0, 4095),
        GridChannel.ACTOR_CLASS: (0, len(ActorKind) - 1),
        GridChannel.ACTOR_TYPE: (0, 4095),
        GridChannel.HEALTH: (0, 32767),
        GridChannel.MAX_HEALTH: (0, 32767),
        GridChannel.ITEM_CLASS: (0, len(ItemKind) - 1),
        GridChannel.ITEM_TYPE: (0, 4095),
        GridChannel.TRAP: (0, len(TrapKind) - 1),
        GridChannel.VISIBILITY: (0, 2),
        GridChannel.STATUS: (0, len(StatusFlag) - 1),
        GridChannel.FACING: (0, 4),
        GridChannel.BEAT_DELAY: (0, 32767),
        GridChannel.BEAT_INTERVAL: (0, 32767),
        GridChannel.FROZEN_TURNS: (0, 32767),
        GridChannel.CONFUSED_TURNS: (0, 32767),
        GridChannel.CHARGE_STATE: (0, 1),
        GridChannel.CHARGE_DIRECTION: (0, 4),
        GridChannel.SHIELD_DIRECTION: (0, 4),
        GridChannel.OBJECT_CLASS: (0, len(ObjectKind) - 1),
        GridChannel.OBJECT_TYPE: (0, 4095),
        GridChannel.INTERACTION_FLAGS: (0, 31),
        GridChannel.PRICE_CURRENCY: (0, 4095),
        GridChannel.PRICE_AMOUNT: (0, 32767),
        GridChannel.PRICE_HEALTH_BP: (0, 32767),
        GridChannel.TRAP_ACTIVATION_DS: (0, 32767),
        GridChannel.TRAP_FAILURE_DS: (0, 32767),
        GridChannel.TELL_ANIMATION_DS: (0, 32767),
        GridChannel.EXPLOSIVE: (0, 1),
    }
    for channel, (low, high) in channel_ranges.items():
        values = grid[..., int(channel)]
        if np.any(values < low) or np.any(values > high):
            raise ProtocolError(
                f"Grid channel {channel.name.lower()!r} contains a value outside {low}..{high}"
            )

    inventory = observation["inventory"]
    if np.any(inventory < 0):
        raise ProtocolError("Inventory values cannot be negative")
    if np.any(inventory[:, 0] >= len(ItemKind)):
        raise ProtocolError("Inventory contains an unknown item identifier")
    if np.any(inventory[:, 1] >= 4096):
        raise ProtocolError("Inventory contains an out-of-range exact type identifier")
    if np.any((inventory[:, 6:8] != 0) & (inventory[:, 6:8] != 1)):
        raise ProtocolError("Inventory ready/active features must be binary")

    mask = observation["action_mask"]
    if np.any((mask != 0) & (mask != 1)):
        raise ProtocolError("The action mask must contain only 0 or 1")

    player = observation["player"]
    for feature in (
        PlayerFeature.HEALTH,
        PlayerFeature.MAX_HEALTH,
        PlayerFeature.GOLD,
        PlayerFeature.GROOVE,
        PlayerFeature.ZONE,
        PlayerFeature.FLOOR,
        PlayerFeature.TURN,
        PlayerFeature.BOMBS,
        PlayerFeature.WEAPON_DAMAGE,
        PlayerFeature.VISIBLE_ENEMIES,
        PlayerFeature.ON_STAIRS,
        PlayerFeature.TASK,
        PlayerFeature.WON,
        PlayerFeature.DEAD,
        PlayerFeature.MUSIC_ELAPSED_DS,
        PlayerFeature.MUSIC_LENGTH_DS,
        PlayerFeature.MUSIC_REMAINING_DS,
        PlayerFeature.SONG_END_REACHED,
        PlayerFeature.SHOP_MUSIC_VOLUME_BP,
    ):
        if player[feature] < 0:
            raise ProtocolError(f"Player feature {feature.name} cannot be negative")
    if player[PlayerFeature.ON_STAIRS] not in {0, 1}:
        raise ProtocolError("Player on-stairs feature must be binary")
    if player[PlayerFeature.WON] not in {0, 1} or player[PlayerFeature.DEAD] not in {0, 1}:
        raise ProtocolError("Player won/dead features must be binary")
    if player[PlayerFeature.SONG_END_REACHED] not in {0, 1}:
        raise ProtocolError("Player song-end feature must be binary")
    return {name: value.copy() for name, value in observation.items()}


def decode_revealed_map(payload: Any) -> np.ndarray | None:
    """Decode an optional spawn-anchored minimap snapshot from Lua."""
    if payload is None:
        return None
    result = _integer_array(
        payload,
        "revealed_map",
        (MAP_SIZE, MAP_SIZE),
        np.dtype(np.int8),
    )
    if np.any(result < 0) or np.any(result > len(Terrain) - 1):
        raise ProtocolError("revealed_map contains an unknown terrain class")
    return result.copy()


def decode_map_bounds(payload: Any) -> dict[str, int] | None:
    """Decode optional absolute level bounds used to detect map-memory clipping."""
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ProtocolError("map_bounds must be an object")
    result = {
        name: _integer(payload.get(name), f"map_bounds.{name}")
        for name in ("x", "y", "width", "height")
    }
    if result["width"] <= 0 or result["height"] <= 0:
        raise ProtocolError("map_bounds width and height must be positive")
    return result


def _validate_events(events: Any) -> None:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise ProtocolError("Record events must be an array")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ProtocolError(f"Event {index} must be an object")
        kind = event.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ProtocolError(f"Event {index} has no valid kind")
        _integer(event.get("amount", 0), f"Event {index} amount", minimum=0)
        if event.get("entity_id") is not None:
            _integer(event["entity_id"], f"Event {index} entity_id", minimum=0)
        data = event.get("data")
        if data is not None and not isinstance(data, Mapping):
            raise ProtocolError(f"Event {index} data must be an object")


def validate_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ProtocolError("An AutoDancer record must be an object")
    if _integer(record.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ProtocolError(
            "Unsupported protocol schema "
            f"{record.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )

    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ProtocolError("Each record must contain a non-empty run_id")
    instance_id = record.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ProtocolError("Each record must contain a non-empty instance_id")
    if not all(character.isalnum() or character in "-_" for character in instance_id):
        raise ProtocolError("instance_id contains unsupported characters")
    if record.get("role") != "worker":
        raise ProtocolError("Turn telemetry must have role 'worker'")
    sequence = _integer(record.get("sequence"), "sequence", minimum=0)
    kind = record.get("kind")
    if kind not in RECORD_KINDS:
        raise ProtocolError(f"Record kind must be one of {sorted(RECORD_KINDS)}")
    status = record.get("episode_status")
    if status not in EPISODE_STATUSES:
        raise ProtocolError(f"episode_status must be one of {sorted(EPISODE_STATUSES)}")

    if kind == "reset" and (sequence != 0 or status != "running"):
        raise ProtocolError("A reset record must be sequence 0 with status 'running'")
    if kind == "turn" and status != "running":
        raise ProtocolError("A turn record must have status 'running'")
    if kind == "terminal" and status == "running":
        raise ProtocolError("A terminal record cannot have status 'running'")

    terminated = record.get("terminated")
    truncated = record.get("truncated")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise ProtocolError("terminated and truncated must be booleans")
    expected_flags = {
        "running": (False, False),
        "won": (True, False),
        "dead": (True, False),
        "aborted": (False, True),
    }[str(status)]
    if (terminated, truncated) != expected_flags:
        raise ProtocolError(
            f"Terminal flags {(terminated, truncated)} do not match status {status!r}"
        )

    if record.get("character") != "Bard":
        raise ProtocolError("AutoDancer-Live-v0 requires Bard")
    game = record.get("game")
    if not isinstance(game, Mapping):
        raise ProtocolError("Record game metadata must be an object")
    version = str(game.get("version", ""))
    steam_build = str(game.get("steam_build", ""))
    if version != SUPPORTED_GAME_VERSION or steam_build != SUPPORTED_STEAM_BUILD:
        raise ProtocolError(
            "Unsupported game build "
            f"{version!r}/{steam_build!r}; expected "
            f"{SUPPORTED_GAME_VERSION!r}/{SUPPORTED_STEAM_BUILD!r}"
        )

    zone = _integer(record.get("zone", 0), "zone", minimum=0)
    floor = _integer(record.get("floor", 0), "floor", minimum=0)
    if status == "running" and (zone < 1 or floor < 1):
        raise ProtocolError("Running records must have positive zone and floor")
    observation = decode_observation(record.get("observation", {}))
    decode_revealed_map(record.get("observation", {}).get("revealed_map"))
    decode_map_bounds(record.get("observation", {}).get("map_bounds"))
    if status == "running":
        mask = observation["action_mask"]
        if not np.all(mask[:5] == 1):
            raise ProtocolError("Running Bard records must enable four directions and WAIT")
        inventory = observation["inventory"]
        expected_specials = np.asarray(
            [
                inventory[6, 0] != 0,
                inventory[1, 0] != 0 and inventory[1, 6] != 0,
                inventory[2, 0] != 0 and inventory[2, 6] != 0,
                inventory[0, 0] != 0,
                inventory[4, 0] != 0 and inventory[4, 6] != 0,
                inventory[5, 0] != 0 and inventory[5, 6] != 0,
            ],
            dtype=np.int8,
        )
        if not np.array_equal(mask[5:], expected_specials):
            raise ProtocolError(
                "Running Bard special-action mask does not match inventory availability"
            )
    _validate_events(record.get("events", []))
    metrics = record.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ProtocolError("Record metrics must be an object")

    bridge = record.get("bridge")
    if bridge is not None:
        if not isinstance(bridge, Mapping):
            raise ProtocolError("Record bridge acknowledgement must be an object")
        session_id = bridge.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ProtocolError("Bridge acknowledgement requires a session_id")
        _integer(bridge.get("command_id"), "bridge command_id", minimum=1)
        bridge_kind = bridge.get("kind")
        if bridge_kind == "ACTION":
            requested = _integer(
                bridge.get("requested_action"), "bridge requested_action", minimum=0
            )
            if requested >= ACTION_COUNT:
                raise ProtocolError(f"bridge requested_action must be below {ACTION_COUNT}")
            engine_action = _integer(
                bridge.get("engine_action"), "bridge engine_action", minimum=0
            )
            observed_action = _integer(
                bridge.get("observed_action"), "bridge observed_action", minimum=0
            )
            if observed_action != engine_action:
                raise ProtocolError("Bridge observed action does not match injected engine action")
        elif bridge_kind == "RESET":
            _integer(bridge.get("seed"), "bridge seed", minimum=0)
        else:
            raise ProtocolError("Bridge acknowledgement kind must be ACTION or RESET")


class TurnSource(Protocol):
    def reset_sequence(self) -> None: ...

    def read(self, timeout: float) -> dict[str, Any]: ...

    def read_latest(self, timeout: float) -> dict[str, Any]: ...


class MessageReceiver(Protocol):
    def receive(self, timeout: float = 10.0, *, max_bytes: int = 65536) -> bytes: ...


class _SequenceTracker:
    def __init__(self) -> None:
        self._expected_sequence: int | None = None
        self._run_id: str | None = None
        self._terminal_seen = False

    def reset_sequence(self) -> None:
        self._expected_sequence = None
        self._run_id = None
        self._terminal_seen = False

    def accept(self, record: dict[str, Any], *, attach: bool = False) -> dict[str, Any]:
        validate_record(record)
        run_id = str(record["run_id"])
        sequence = int(record["sequence"])
        kind = str(record["kind"])

        if attach and self._run_id is None:
            self._run_id = run_id
            self._expected_sequence = sequence + 1
            self._terminal_seen = kind == "terminal"
            return record

        if kind == "reset":
            self._run_id = run_id
            self._expected_sequence = 1
            self._terminal_seen = False
            return record

        if self._run_id is None or self._expected_sequence is None:
            raise ProtocolError("A turn source must start with a reset record or attach explicitly")
        if self._terminal_seen:
            raise ProtocolError("A record arrived after the run's terminal record")
        if run_id != self._run_id:
            raise ProtocolError(
                f"Run identity changed from {self._run_id!r} to {run_id!r} without a reset"
            )
        if sequence != self._expected_sequence:
            raise ProtocolError(
                f"Turn sequence mismatch: expected {self._expected_sequence}, received {sequence}"
            )
        self._expected_sequence = sequence + 1
        self._terminal_seen = kind == "terminal"
        return record


class JsonlTurnSource(_SequenceTracker):
    """Tail records that the local Lua mod prints to the game debug log."""

    def __init__(self, path: str | Path, *, start_at_end: bool = True) -> None:
        super().__init__()
        self.path = Path(path)
        self.start_at_end = start_at_end
        self._offset: int | None = None

    def reset_sequence(self) -> None:
        super().reset_sequence()
        # Establish the boundary before LiveEnv publishes a restart command so a fast
        # reset record cannot be skipped by lazy start-at-end initialization.
        if self._offset is None and self.start_at_end:
            self._offset = self.path.stat().st_size if self.path.exists() else 0

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
                        return self.accept(_decode_record_line(line, marker_index))
            time.sleep(0.01)
        raise TimeoutError(f"No AutoDancer turn record arrived within {timeout:.1f} seconds")

    def read_latest(self, timeout: float = 5.0) -> dict[str, Any]:
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
                    return self.accept(latest, attach=True)
            time.sleep(0.01)
        raise TimeoutError(
            f"No existing AutoDancer turn record was found within {timeout:.1f} seconds"
        )


class NativePipeTurnSource(_SequenceTracker):
    """Read schema-9 JSON messages directly from a worker's duplex pipe."""

    def __init__(self, receiver: MessageReceiver) -> None:
        super().__init__()
        self.receiver = receiver

    def read(self, timeout: float = 5.0) -> dict[str, Any]:
        payload = self.receiver.receive(timeout, max_bytes=65536)
        try:
            record = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError("An AutoDancer pipe record contains invalid JSON") from error
        if not isinstance(record, dict):
            raise ProtocolError("An AutoDancer pipe record must be a JSON object")
        return self.accept(record)

    def read_latest(self, timeout: float = 5.0) -> dict[str, Any]:
        record = self.read(timeout)
        self.reset_sequence()
        return self.accept(record, attach=True)


class QueueTurnSource(_SequenceTracker):
    """In-memory source for tests and prerecorded live sessions."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        super().__init__()
        self.records = deque(records)

    def read(self, timeout: float = 5.0) -> dict[str, Any]:
        del timeout
        if not self.records:
            raise TimeoutError("The in-memory turn source is empty")
        return self.accept(self.records.popleft())

    def read_latest(self, timeout: float = 5.0) -> dict[str, Any]:
        del timeout
        if not self.records:
            raise TimeoutError("The in-memory turn source is empty")
        return self.accept(self.records.popleft(), attach=True)
