"""Lineage-bound successful action traces for legal live-game replay."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autodancer.constants import ACTION_COUNT, PlayerFeature

DEMONSTRATION_BANK_SCHEMA = 1
SUCCESS_STATUSES = frozenset({"curriculum_complete", "victory"})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def normalized_observation_digest(observation: Mapping[str, Any]) -> str:
    """Hash policy observations while excluding wall-clock-derived music fields."""

    digest = hashlib.sha256()
    for name in sorted(observation):
        value = np.asarray(observation[name]).copy()
        if name == "player":
            value[[PlayerFeature.MUSIC_ELAPSED_DS, PlayerFeature.MUSIC_REMAINING_DS]] = 0
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(_canonical_json(contiguous.shape))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_demonstration_sources(payload: Mapping[str, Any]) -> None:
    """Require every source journal to retain the bytes bound into the bank."""

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("demonstration bank sources must be a non-empty list")
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("demonstration bank source must be an object")
        path = Path(str(source.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"demonstration source no longer exists: {path}")
        if _sha256_file(path) != str(source.get("sha256", "")):
            raise ValueError(f"demonstration source hash mismatch: {path}")


def _integer_mapping(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"successful episode {field} must be an object")
    result = {str(key): int(count) for key, count in value.items()}
    if any(count < 0 for count in result.values()):
        raise ValueError(f"successful episode {field} counts must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class SuccessfulActionTrace:
    """One reproducible candidate trace extracted from a successful live episode."""

    trace_id: str
    seed: int
    source_run_id: str
    source_worker_id: str
    source_policy_version: int
    source_global_step: int
    status: str
    turns: int
    furthest_zone: int
    furthest_floor: int
    curriculum_reset: dict[str, Any]
    boss_progress: dict[str, Any]
    event_counts: dict[str, int]
    action_sequence: tuple[int, ...]

    @classmethod
    def from_episode(cls, episode: Mapping[str, Any]) -> SuccessfulActionTrace | None:
        """Return a trace for a valid success, or ``None`` for an ordinary failure."""

        status = str(episode.get("status", ""))
        if status not in SUCCESS_STATUSES:
            return None
        if episode.get("natural_prefix") or episode.get("learning_segment"):
            # Collector episode actions begin at the learner handoff. Such a
            # sequence is not replayable from the recorded curriculum reset.
            return None
        if episode.get("infrastructure_valid") is not True:
            raise ValueError("successful episode is not infrastructure-valid")
        actions = episode.get("successful_action_sequence")
        if not isinstance(actions, list) or not actions:
            raise ValueError("successful episode has no action sequence")
        sequence = tuple(int(action) for action in actions)
        if any(action < 0 or action >= ACTION_COUNT for action in sequence):
            raise ValueError("successful episode contains an invalid action")
        turns = int(episode.get("turns", -1))
        if turns != len(sequence):
            raise ValueError(
                "successful episode turn count does not match its action sequence: "
                f"turns={turns}, actions={len(sequence)}"
            )
        curriculum_reset = episode.get("curriculum_reset")
        boss_progress = episode.get("boss_progress")
        if not isinstance(curriculum_reset, Mapping):
            raise ValueError("successful episode curriculum_reset must be an object")
        if not isinstance(boss_progress, Mapping):
            raise ValueError("successful episode boss_progress must be an object")
        core = {
            "seed": int(episode["seed"]),
            "source_run_id": str(episode.get("run_id", "")),
            "source_worker_id": str(episode.get("worker_id", "")),
            "source_policy_version": int(episode.get("policy_version", -1)),
            "source_global_step": int(episode.get("global_step", -1)),
            "status": status,
            "turns": turns,
            "furthest_zone": int(episode.get("furthest_zone", 0)),
            "furthest_floor": int(episode.get("furthest_floor", 0)),
            "curriculum_reset": dict(curriculum_reset),
            "boss_progress": dict(boss_progress),
            "event_counts": _integer_mapping(episode.get("event_counts", {}), "event_counts"),
            "action_sequence": sequence,
        }
        if not core["source_run_id"] or not core["source_worker_id"]:
            raise ValueError("successful episode is missing run or worker identity")
        if core["source_policy_version"] < 0 or core["source_global_step"] < 0:
            raise ValueError("successful episode is missing policy or step identity")
        trace_id = hashlib.sha256(_canonical_json(core)).hexdigest()
        return cls(trace_id=trace_id, **core)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "seed": self.seed,
            "source_run_id": self.source_run_id,
            "source_worker_id": self.source_worker_id,
            "source_policy_version": self.source_policy_version,
            "source_global_step": self.source_global_step,
            "status": self.status,
            "turns": self.turns,
            "furthest_zone": self.furthest_zone,
            "furthest_floor": self.furthest_floor,
            "curriculum_reset": self.curriculum_reset,
            "boss_progress": self.boss_progress,
            "event_counts": self.event_counts,
            "action_sequence": list(self.action_sequence),
        }


def load_successful_episode_traces(path: str | Path) -> tuple[SuccessfulActionTrace, ...]:
    """Extract all valid successful traces from one append-only episode journal."""

    source = Path(path)
    traces: list[SuccessfulActionTrace] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed episode JSON at {source}:{line_number}") from error
            if not isinstance(episode, Mapping):
                raise ValueError(f"episode record at {source}:{line_number} must be an object")
            try:
                trace = SuccessfulActionTrace.from_episode(episode)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid successful episode at {source}:{line_number}: {error}"
                ) from error
            if trace is not None:
                traces.append(trace)
    return tuple(traces)


def load_successful_evaluation_traces(path: str | Path) -> tuple[SuccessfulActionTrace, ...]:
    """Extract replayable successes from a trained-policy live evaluation report."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("trained"), Mapping):
        raise ValueError("evaluation trace source must contain a trained policy report")
    results = payload["trained"].get("results")
    if not isinstance(results, list):
        raise ValueError("evaluation trace source has no trained episode results")
    reset = {
        "id": "evaluation",
        "start_level": int(payload.get("curriculum_start_level", 1)),
        "target_level": payload.get("curriculum_target_level"),
        "profile": str(payload.get("curriculum_profile", "normal")),
    }
    traces: list[SuccessfulActionTrace] = []
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise ValueError(f"evaluation result {index} must be an object")
        episode = {
            "seed": result.get("seed"),
            "status": result.get("status"),
            "infrastructure_valid": payload.get("controller_valid") is True,
            "run_id": result.get("run_id"),
            "worker_id": result.get("worker_id"),
            "policy_version": int(payload.get("checkpoint_updates", -1)),
            "global_step": int(payload.get("checkpoint_global_step", -1)),
            "turns": result.get("turns"),
            "furthest_zone": result.get("furthest_zone"),
            "furthest_floor": result.get("furthest_floor"),
            "curriculum_reset": reset,
            "boss_progress": {
                "boss_type": result.get("boss_type"),
                "initial_health": result.get("initial_boss_health"),
                "minimum_health": result.get("minimum_boss_health"),
                "boss_damage": result.get("boss_damage"),
                "observed_actor_types": result.get("boss_actor_types", []),
                "reached": result.get("death_metal_phase4_reached", False),
            },
            "event_counts": result.get("event_counts", {}),
            "successful_action_sequence": result.get("successful_action_sequence"),
            "natural_prefix": result.get("natural_prefix", {}),
        }
        try:
            trace = SuccessfulActionTrace.from_episode(episode)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid successful evaluation result {index}: {error}") from error
        if trace is not None:
            traces.append(trace)
    return tuple(traces)


def load_successful_traces(path: str | Path) -> tuple[SuccessfulActionTrace, ...]:
    """Load either an append-only training journal or a baseline report."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return load_successful_episode_traces(source)
    if isinstance(payload, Mapping) and "trained" in payload:
        return load_successful_evaluation_traces(source)
    return load_successful_episode_traces(source)


def build_demonstration_bank(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Build a deterministic trace bank with source hashes and duplicate removal."""

    if not paths:
        raise ValueError("at least one episode journal is required")
    sources: list[dict[str, Any]] = []
    unique: dict[tuple[int, tuple[int, ...]], SuccessfulActionTrace] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"episode journal does not exist: {path}")
        traces = load_successful_traces(path)
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "kind": (
                    "evaluation-report" if path.suffix.lower() == ".json" else "episode-journal"
                ),
                "successful_trace_count": len(traces),
            }
        )
        for trace in traces:
            unique.setdefault((trace.seed, trace.action_sequence), trace)
    ordered = sorted(unique.values(), key=lambda trace: (trace.seed, trace.trace_id))
    payload: dict[str, Any] = {
        "schema_version": DEMONSTRATION_BANK_SCHEMA,
        "kind": "qualified-live-action-traces-v1",
        "sources": sources,
        "traces": [trace.as_dict() for trace in ordered],
    }
    payload["bank_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def validate_demonstration_bank(payload: Mapping[str, Any]) -> None:
    """Validate schema, bank identity, and every embedded successful trace."""

    expected_keys = {"schema_version", "kind", "sources", "traces", "bank_sha256"}
    if set(payload) != expected_keys:
        raise ValueError("demonstration bank has unexpected fields")
    if payload.get("schema_version") != DEMONSTRATION_BANK_SCHEMA:
        raise ValueError("unsupported demonstration bank schema")
    if payload.get("kind") != "qualified-live-action-traces-v1":
        raise ValueError("unsupported demonstration bank kind")
    expected_hash = str(payload.get("bank_sha256", ""))
    unhashed = {key: value for key, value in payload.items() if key != "bank_sha256"}
    actual_hash = hashlib.sha256(_canonical_json(unhashed)).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError("demonstration bank hash mismatch")
    traces = payload.get("traces")
    if not isinstance(traces, list):
        raise ValueError("demonstration bank traces must be a list")
    seen_ids: set[str] = set()
    seen_sequences: set[tuple[int, tuple[int, ...]]] = set()
    for raw_trace in traces:
        if not isinstance(raw_trace, Mapping):
            raise ValueError("demonstration trace must be an object")
        episode = dict(raw_trace)
        episode["infrastructure_valid"] = True
        episode["successful_action_sequence"] = episode.pop("action_sequence", None)
        episode["run_id"] = episode.pop("source_run_id", "")
        episode["worker_id"] = episode.pop("source_worker_id", "")
        episode["policy_version"] = episode.pop("source_policy_version", -1)
        episode["global_step"] = episode.pop("source_global_step", -1)
        expected_trace_id = str(episode.pop("trace_id", ""))
        trace = SuccessfulActionTrace.from_episode(episode)
        if trace is None or trace.trace_id != expected_trace_id:
            raise ValueError("demonstration trace identity mismatch")
        key = (trace.seed, trace.action_sequence)
        if trace.trace_id in seen_ids or key in seen_sequences:
            raise ValueError("demonstration bank contains duplicate traces")
        seen_ids.add(trace.trace_id)
        seen_sequences.add(key)


def write_demonstration_bank(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Validate and atomically write a demonstration bank."""

    validate_demonstration_bank(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def iter_trace_actions(trace: Mapping[str, Any]) -> Iterable[int]:
    """Yield a validated trace action sequence for a future live replay driver."""

    actions = trace.get("action_sequence")
    if not isinstance(actions, list) or not actions:
        raise ValueError("demonstration trace has no action sequence")
    for action in actions:
        selected = int(action)
        if selected < 0 or selected >= ACTION_COUNT:
            raise ValueError("demonstration trace contains an invalid action")
        yield selected
