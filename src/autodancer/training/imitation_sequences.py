"""Hash-bound recurrent demonstration sequences for actor-only imitation.

The artifact is deliberately separate from PPO rollouts.  It stores the exact
policy inputs observed during a freshly qualified live replay, plus the action
and feedback context needed to reconstruct an LSTM state.  Nothing in this
module enables imitation during ordinary training; experiments must opt in with
an explicit, lineage-bound objective.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autodancer.constants import ACTION_COUNT
from autodancer.observation import observation_space
from autodancer.training.model import START_ACTION

RECURRENT_DEMONSTRATION_SCHEMA = 1
OBSERVATION_NAMES = ("grid", "map_memory", "player", "inventory", "action_mask")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path: str | Path) -> Path:
    artifact = Path(path)
    return artifact.with_name(f"{artifact.name}.manifest.json")


@dataclass(frozen=True, slots=True)
class RecurrentDemonstration:
    """One ordered successful sequence of policy inputs and target actions."""

    trace_id: str
    seed: int
    observations: Mapping[str, np.ndarray]
    actions: np.ndarray
    previous_actions: np.ndarray
    previous_rewards: np.ndarray
    episode_starts: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    def validate(self) -> None:
        if not self.trace_id:
            raise ValueError("recurrent demonstration has no trace identity")
        if self.seed < 0:
            raise ValueError("recurrent demonstration seed must be non-negative")
        if set(self.observations) != set(OBSERVATION_NAMES):
            raise ValueError("recurrent demonstration observation keys do not match the policy")
        if self.actions.ndim != 1 or self.actions.size == 0:
            raise ValueError("recurrent demonstration actions must be a non-empty vector")
        length = self.length
        vectors = {
            "previous_actions": self.previous_actions,
            "previous_rewards": self.previous_rewards,
            "episode_starts": self.episode_starts,
        }
        if any(value.shape != (length,) for value in vectors.values()):
            raise ValueError("recurrent demonstration context length mismatch")
        actions = np.asarray(self.actions)
        previous_actions = np.asarray(self.previous_actions)
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("recurrent demonstration actions must be integers")
        if np.any((actions < 0) | (actions >= ACTION_COUNT)):
            raise ValueError("recurrent demonstration contains an invalid action")
        if not np.issubdtype(previous_actions.dtype, np.integer):
            raise ValueError("recurrent demonstration previous actions must be integers")
        if np.any((previous_actions < 0) | (previous_actions > START_ACTION)):
            raise ValueError("recurrent demonstration contains invalid previous-action context")
        expected_previous = np.concatenate(
            (np.asarray([START_ACTION], dtype=previous_actions.dtype), actions[:-1])
        )
        if not np.array_equal(previous_actions, expected_previous):
            raise ValueError("recurrent demonstration previous actions are not contiguous")
        rewards = np.asarray(self.previous_rewards)
        if not np.issubdtype(rewards.dtype, np.floating) or not np.all(np.isfinite(rewards)):
            raise ValueError("recurrent demonstration feedback must be finite floating point")
        starts = np.asarray(self.episode_starts)
        if starts.dtype != np.bool_ or not bool(starts[0]) or np.any(starts[1:]):
            raise ValueError("recurrent demonstration must contain one leading episode boundary")

        spaces = observation_space().spaces
        for name in OBSERVATION_NAMES:
            value = np.asarray(self.observations[name])
            expected = spaces[name]
            if value.shape != (length, *expected.shape):
                raise ValueError(
                    f"recurrent demonstration {name} has shape {value.shape}; "
                    f"expected {(length, *expected.shape)}"
                )
            if value.dtype != expected.dtype:
                raise ValueError(
                    f"recurrent demonstration {name} has dtype {value.dtype}; "
                    f"expected {expected.dtype}"
                )
        masks = np.asarray(self.observations["action_mask"])
        if np.any((masks != 0) & (masks != 1)):
            raise ValueError("recurrent demonstration action masks must be binary")
        if not np.all(masks[np.arange(length), actions.astype(np.int64)] == 1):
            raise ValueError("recurrent demonstration selects a masked action")


def _array_key(index: int, name: str) -> str:
    return f"trace_{index:04d}_{name}"


def write_recurrent_demonstrations(
    path: str | Path,
    demonstrations: Sequence[RecurrentDemonstration],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically write compressed arrays and a hash-bound JSON manifest."""

    if not demonstrations:
        raise ValueError("at least one recurrent demonstration is required")
    seen: set[str] = set()
    arrays: dict[str, np.ndarray] = {}
    trace_metadata: list[dict[str, Any]] = []
    for index, demonstration in enumerate(demonstrations):
        demonstration.validate()
        if demonstration.trace_id in seen:
            raise ValueError("recurrent demonstration trace identities must be unique")
        seen.add(demonstration.trace_id)
        names: dict[str, str] = {}
        for name in OBSERVATION_NAMES:
            key = _array_key(index, name)
            arrays[key] = np.ascontiguousarray(demonstration.observations[name])
            names[name] = key
        for name, value in (
            ("actions", demonstration.actions),
            ("previous_actions", demonstration.previous_actions),
            ("previous_rewards", demonstration.previous_rewards),
            ("episode_starts", demonstration.episode_starts),
        ):
            key = _array_key(index, name)
            arrays[key] = np.ascontiguousarray(value)
            names[name] = key
        trace_metadata.append(
            {
                "trace_id": demonstration.trace_id,
                "seed": demonstration.seed,
                "length": demonstration.length,
                "arrays": names,
            }
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    manifest: dict[str, Any] = {
        "schema_version": RECURRENT_DEMONSTRATION_SCHEMA,
        "kind": "qualified-live-recurrent-demonstrations-v1",
        "artifact": str(destination.resolve()),
        "artifact_sha256": _sha256_file(destination),
        "provenance": dict(provenance),
        "traces": trace_metadata,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    sidecar = manifest_path(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, sidecar)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return manifest


def load_recurrent_demonstrations(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[RecurrentDemonstration, ...]]:
    """Validate all hashes, shapes, context, and action masks before returning data."""

    artifact = Path(path).resolve()
    sidecar = manifest_path(artifact)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("recurrent demonstration manifest must be an object")
    expected_keys = {
        "schema_version",
        "kind",
        "artifact",
        "artifact_sha256",
        "provenance",
        "traces",
        "manifest_sha256",
    }
    if set(manifest) != expected_keys:
        raise ValueError("recurrent demonstration manifest has unexpected fields")
    if manifest["schema_version"] != RECURRENT_DEMONSTRATION_SCHEMA:
        raise ValueError("unsupported recurrent demonstration schema")
    if manifest["kind"] != "qualified-live-recurrent-demonstrations-v1":
        raise ValueError("unsupported recurrent demonstration kind")
    expected_manifest_hash = str(manifest["manifest_sha256"])
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical_json(unhashed)).hexdigest() != expected_manifest_hash:
        raise ValueError("recurrent demonstration manifest hash mismatch")
    if Path(str(manifest["artifact"])).resolve() != artifact:
        raise ValueError("recurrent demonstration artifact path mismatch")
    if _sha256_file(artifact) != str(manifest["artifact_sha256"]):
        raise ValueError("recurrent demonstration artifact hash mismatch")
    raw_traces = manifest["traces"]
    if not isinstance(raw_traces, list) or not raw_traces:
        raise ValueError("recurrent demonstration manifest contains no traces")

    loaded: list[RecurrentDemonstration] = []
    seen_arrays: set[str] = set()
    with np.load(artifact, allow_pickle=False) as arrays:
        available = set(arrays.files)
        for raw in raw_traces:
            if not isinstance(raw, dict) or set(raw) != {"trace_id", "seed", "length", "arrays"}:
                raise ValueError("invalid recurrent demonstration trace manifest")
            names = raw["arrays"]
            required = {
                *OBSERVATION_NAMES,
                "actions",
                "previous_actions",
                "previous_rewards",
                "episode_starts",
            }
            if not isinstance(names, dict) or set(names) != required:
                raise ValueError("recurrent demonstration array mapping is incomplete")
            keys = {str(value) for value in names.values()}
            if len(keys) != len(required) or not keys <= available or keys & seen_arrays:
                raise ValueError("recurrent demonstration array mapping is invalid")
            seen_arrays.update(keys)
            observation = {name: arrays[str(names[name])].copy() for name in OBSERVATION_NAMES}
            demonstration = RecurrentDemonstration(
                trace_id=str(raw["trace_id"]),
                seed=int(raw["seed"]),
                observations=observation,
                actions=arrays[str(names["actions"])].copy(),
                previous_actions=arrays[str(names["previous_actions"])].copy(),
                previous_rewards=arrays[str(names["previous_rewards"])].copy(),
                episode_starts=arrays[str(names["episode_starts"])].copy(),
            )
            demonstration.validate()
            if demonstration.length != int(raw["length"]):
                raise ValueError("recurrent demonstration manifest length mismatch")
            loaded.append(demonstration)
        if seen_arrays != available:
            raise ValueError("recurrent demonstration artifact contains unreferenced arrays")
    if len({trace.trace_id for trace in loaded}) != len(loaded):
        raise ValueError("recurrent demonstration manifest repeats a trace identity")
    return manifest, tuple(loaded)
