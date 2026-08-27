"""Audit schema-10 type-name hashes for within-channel observation aliases."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

TYPE_HASH_MODULUS = 4095


def lua_type_id(name: str) -> int:
    """Reproduce ``AutoDancer.lua``'s stable nonzero type identifier exactly."""

    value = 0
    # Synchrony's Lua runtime lowercases internal ASCII type names byte-wise.
    # Applying the conversion ourselves avoids Python Unicode case folding
    # changing the byte sequence for a hypothetical non-ASCII name.
    for byte in name.encode("utf-8"):
        if 65 <= byte <= 90:
            byte += 32
        value = (value * 31 + byte) % TYPE_HASH_MODULUS
    return value + 1


def _load_catalog(path: Path) -> dict[str, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read type catalog {path}: {error}") from error
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Type catalog must use schema_version 1")
    channels = payload.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise ValueError("Type catalog must contain a nonempty channels object")
    result: dict[str, list[str]] = {}
    for channel, names in channels.items():
        if not isinstance(channel, str) or not channel:
            raise ValueError("Type catalog channel names must be nonempty strings")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"Type catalog channel {channel!r} must contain strings")
        normalized = [name for name in names if name]
        if len({name.lower() for name in normalized}) != len(normalized):
            raise ValueError(f"Type catalog channel {channel!r} contains duplicate names")
        result[channel] = normalized
    return result


def audit_catalog(channels: dict[str, list[str]]) -> dict[str, Any]:
    """Report only aliases that share an embedding table/semantic channel."""

    channel_reports: dict[str, Any] = {}
    total_names = 0
    total_collision_groups = 0
    total_colliding_names = 0
    for channel in sorted(channels):
        names = channels[channel]
        identifiers: defaultdict[int, list[str]] = defaultdict(list)
        for name in names:
            identifiers[lua_type_id(name)].append(name)
        collisions = [
            {"type_id": identifier, "names": sorted(group, key=str.lower)}
            for identifier, group in sorted(identifiers.items())
            if len(group) > 1
        ]
        colliding_names = sum(len(collision["names"]) for collision in collisions)
        channel_reports[channel] = {
            "names": len(names),
            "unique_ids": len(identifiers),
            "collision_groups": len(collisions),
            "colliding_names": colliding_names,
            "collisions": collisions,
        }
        total_names += len(names)
        total_collision_groups += len(collisions)
        total_colliding_names += colliding_names
    return {
        "schema_version": 1,
        "hash": {"algorithm": "lowercase-polynomial-31", "modulus": TYPE_HASH_MODULUS},
        "channels": channel_reports,
        "summary": {
            "names": total_names,
            "collision_groups": total_collision_groups,
            "colliding_names": total_colliding_names,
            "collision_free": total_collision_groups == 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AutoDancer type identity collisions")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit_catalog(_load_catalog(arguments.catalog))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(arguments.output)
    print(rendered)
    return 0 if report["summary"]["collision_free"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
