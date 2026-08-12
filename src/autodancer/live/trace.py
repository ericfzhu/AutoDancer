"""Record format and simulator comparison for live conformance traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autodancer.envs.sim import AutoDancerSimEnv

TRACE_SCHEMA_VERSION = 1


def load_trace(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("kind") != "header":
        raise ValueError("A conformance trace must start with a header record")
    header, turns = records[0], records[1:]
    if header.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError("Unsupported conformance trace schema")
    if not header.get("game_version") or not header.get("steam_build"):
        raise ValueError("Trace header must include game_version and steam_build")
    for expected_sequence, turn in enumerate(turns, start=1):
        if turn.get("kind") != "turn" or turn.get("sequence") != expected_sequence:
            raise ValueError(f"Invalid trace turn at sequence {expected_sequence}")
    return header, turns


def _compare(
    expected: Any,
    actual: Any,
    path: str,
    differences: list[str],
    ignored: set[str],
) -> None:
    if path in ignored:
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in expected.keys() | actual.keys():
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append(f"{child}: unexpected value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child}: missing; expected {expected[key]!r}")
            else:
                _compare(expected[key], actual[key], child, differences, ignored)
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            differences.append(f"{path}: expected a sequence, got {type(actual).__name__}")
        elif len(expected) != len(actual):
            differences.append(f"{path}: length {len(actual)}; expected {len(expected)}")
        else:
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                _compare(left, right, f"{path}[{index}]", differences, ignored)
    elif expected != actual:
        differences.append(f"{path}: {actual!r}; expected {expected!r}")


def compare_trace(path: str | Path) -> list[str]:
    header, turns = load_trace(path)
    environment = AutoDancerSimEnv(task=header["task"])
    environment.reset(
        seed=int(header["seed"]),
        options={"zone": header.get("zone", 1), "floor": header.get("floor", 1)},
    )
    differences: list[str] = []
    ignored = set(header.get("ignored_paths", []))
    for turn in turns:
        _, _, terminated, truncated, _ = environment.step(int(turn["action"]))
        turn_differences: list[str] = []
        _compare(turn["state"], environment.snapshot(), "", turn_differences, ignored)
        differences.extend(
            f"turn {turn['sequence']}: {difference}" for difference in turn_differences
        )
        if terminated or truncated:
            break
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a live trace with the simulator")
    parser.add_argument("trace", type=Path)
    arguments = parser.parse_args()
    differences = compare_trace(arguments.trace)
    if differences:
        for difference in differences:
            print(difference)
        raise SystemExit(1)
    print(f"Trace conforms: {arguments.trace}")


if __name__ == "__main__":
    main()

