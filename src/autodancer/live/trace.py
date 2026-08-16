"""Versioned live trace recording and simulator comparison."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autodancer.envs.sim import AutoDancerSimEnv
from autodancer.live.protocol import SCHEMA_VERSION as PROTOCOL_SCHEMA_VERSION

TRACE_SCHEMA_VERSION = 2
SUPPORTED_TRACE_SCHEMAS = frozenset({1, TRACE_SCHEMA_VERSION})


def serialize_observation(observation: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {name: np.asarray(value).tolist() for name, value in observation.items()}


def _write_jsonl(path: Path, record: Mapping[str, Any], *, append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


class TraceWriter:
    """Write one action and its matching post-action live record per line."""

    def __init__(
        self,
        path: str | Path,
        *,
        info: Mapping[str, Any],
        task: str,
        initial_observation: Mapping[str, np.ndarray] | None = None,
        ignored_paths: Sequence[str] = (),
        strict: bool = False,
        compare_observation: bool = False,
        compare_reward: bool = False,
        compare_events: bool = False,
        compare_state: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(
                f"Trace already exists: {self.path}. Pass overwrite=True to replace it."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self.compare_observation = compare_observation
        self.compare_reward = compare_reward
        self.compare_events = compare_events
        self.compare_state = compare_state
        game = info.get("game", {})
        header: dict[str, Any] = {
            "kind": "header",
            "schema_version": TRACE_SCHEMA_VERSION,
            "protocol_schema_version": int(
                info.get("protocol_schema_version", PROTOCOL_SCHEMA_VERSION)
            ),
            "game_version": game.get("version"),
            "steam_build": game.get("steam_build"),
            "run_id": info.get("run_id"),
            "seed": info.get("seed"),
            "task": task,
            "zone": info.get("zone", 1),
            "floor": info.get("floor", 1),
            "ignored_paths": list(ignored_paths),
            "strict": bool(strict),
            "compare_observation": bool(compare_observation),
            "compare_reward": bool(compare_reward),
            "compare_events": bool(compare_events),
            "compare_state": bool(compare_state),
        }
        if initial_observation is not None:
            serialized = serialize_observation(initial_observation)
            header["initial_live_observation"] = serialized
            if compare_observation:
                header["initial_observation"] = serialized
        if not header["game_version"] or not header["steam_build"]:
            raise ValueError("Trace metadata must pin game_version and steam_build")
        if not header["run_id"]:
            raise ValueError("Trace metadata must include a run_id")
        if header["seed"] is None:
            raise ValueError("Trace metadata must include a seed")
        _write_jsonl(self.path, header, append=False)

    def append(
        self,
        *,
        action: int,
        observation: Mapping[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self._sequence += 1
        live_observation = serialize_observation(observation)
        record: dict[str, Any] = {
            "kind": "turn",
            "sequence": self._sequence,
            "live_sequence": info.get("sequence"),
            "action": int(action),
            "live_observation": live_observation,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_status": info.get("episode_status"),
        }
        if self.compare_observation:
            record["observation"] = live_observation
        if self.compare_reward:
            record["reward"] = float(reward)
        if self.compare_events:
            record["events"] = info.get("raw_events", [])
        if self.compare_state and state is not None:
            record["state"] = state
        _write_jsonl(self.path, record, append=True)


def load_trace(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("kind") != "header":
        raise ValueError("A conformance trace must start with a header record")
    header, turns = records[0], records[1:]
    schema = int(header.get("schema_version", -1))
    if schema not in SUPPORTED_TRACE_SCHEMAS:
        raise ValueError(f"Unsupported conformance trace schema {schema}")
    if not header.get("game_version") or not header.get("steam_build"):
        raise ValueError("Trace header must include game_version and steam_build")
    if "seed" not in header or "task" not in header:
        raise ValueError("Trace header must include seed and task")
    for expected_sequence, turn in enumerate(turns, start=1):
        if turn.get("kind") != "turn" or turn.get("sequence") != expected_sequence:
            raise ValueError(f"Invalid trace turn at sequence {expected_sequence}")
        if "action" not in turn:
            raise ValueError(f"Trace turn {expected_sequence} has no action")
    return header, turns


def _ignored(path: str, patterns: set[str]) -> bool:
    return any(path == pattern or fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _compare(
    expected: Any,
    actual: Any,
    path: str,
    differences: list[str],
    ignored: set[str],
    *,
    strict: bool,
) -> None:
    if _ignored(path, ignored):
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            differences.append(
                f"{path or '<root>'}: expected an object, "
                f"got {type(actual).__name__}"
            )
            return
        keys = expected.keys() | actual.keys() if strict else expected.keys()
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append(f"{child}: unexpected value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child}: missing; expected {expected[key]!r}")
            else:
                _compare(expected[key], actual[key], child, differences, ignored, strict=strict)
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            differences.append(f"{path}: expected a sequence, got {type(actual).__name__}")
        elif len(expected) != len(actual):
            differences.append(f"{path}: length {len(actual)}; expected {len(expected)}")
        else:
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                _compare(
                    left,
                    right,
                    f"{path}[{index}]",
                    differences,
                    ignored,
                    strict=strict,
                )
    elif isinstance(expected, float) or isinstance(actual, float):
        try:
            equal = math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            equal = False
        if not equal:
            differences.append(f"{path}: {actual!r}; expected {expected!r}")
    elif expected != actual:
        differences.append(f"{path}: {actual!r}; expected {expected!r}")


def compare_trace(path: str | Path, *, strict_override: bool = False) -> list[str]:
    header, turns = load_trace(path)
    environment = AutoDancerSimEnv(task=header["task"])
    observation, _ = environment.reset(
        seed=int(header["seed"]),
        options={"zone": header.get("zone", 1), "floor": header.get("floor", 1)},
    )
    differences: list[str] = []
    ignored = set(str(item) for item in header.get("ignored_paths", []))
    schema = int(header["schema_version"])
    strict = bool(header.get("strict", False) or strict_override)

    if schema == TRACE_SCHEMA_VERSION and "initial_observation" in header:
        initial_differences: list[str] = []
        _compare(
            header["initial_observation"],
            serialize_observation(observation),
            "observation",
            initial_differences,
            ignored,
            strict=strict,
        )
        differences.extend(f"initial: {item}" for item in initial_differences)

    for turn in turns:
        observation, reward, terminated, truncated, info = environment.step(int(turn["action"]))
        actual = {
            "state": environment.snapshot(),
            "observation": serialize_observation(observation),
            "reward": float(reward),
            "events": info.get("raw_events", []),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_status": info.get("episode_status"),
        }
        turn_differences: list[str] = []
        if schema == 1:
            _compare(
                turn["state"], actual["state"], "", turn_differences, ignored, strict=False
            )
        else:
            expected = {
                key: turn[key]
                for key in (
                    "state",
                    "observation",
                    "reward",
                    "events",
                    "terminated",
                    "truncated",
                    "episode_status",
                )
                if key in turn
            }
            _compare(expected, actual, "", turn_differences, ignored, strict=strict)
        differences.extend(
            f"turn {turn['sequence']}: {difference}" for difference in turn_differences
        )
        if terminated or truncated:
            if turn["sequence"] != len(turns):
                differences.append(
                    f"turn {turn['sequence']}: simulator ended before the trace's final turn"
                )
            break
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a live trace with the simulator")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()
    differences = compare_trace(arguments.trace, strict_override=arguments.strict)
    if differences:
        for difference in differences:
            print(difference)
        raise SystemExit(1)
    print(f"Trace conforms: {arguments.trace}")


if __name__ == "__main__":
    main()
