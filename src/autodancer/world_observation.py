"""Versioned, collision-free world records; no aggregation of co-located entities."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from gymnasium import spaces


@lru_cache(maxsize=1)
def world_registry():
    raw = Path(__file__).with_name("data").joinpath("world_registry.json").read_bytes()
    data = json.loads(raw)
    for key in ("entities", "tiles", "currencies", "markers"):
        if len(set(data[key])) != len(data[key]) or not all(data[key]):
            raise ValueError("Invalid world registry")
    data["entity_ids"] = {name: i + 1 for i, name in enumerate(data["entities"])}
    data["tile_ids"] = {name: i + 1 for i, name in enumerate(data["tiles"])}
    return data, hashlib.sha256(raw).hexdigest()


FIELDS = tuple(".".join(pair) for pair in world_registry()[0]["fields"])
MARKERS = tuple(world_registry()[0]["markers"])
STATUSES = ("freezable", "confusable", "barrier", "invincibility")
FACTS = (*FIELDS, *("marker." + m for m in MARKERS), *(s + ".active" for s in STATUSES))
# Conservative cues only. Exact AI phases and timers are internal diagnostics.
# A full human-equivalent rendering contract remains separate from this schema.
PLAYER_FIELDS = frozenset(
    {
        "facingDirection.direction",
        "shieldDirection.direction",
        "charge.active",
        "charge.direction",
        "interactable.active",
        "shrine.active",
        "priceTag.active",
        "priceTagCostCurrency.cost",
        "priceTagCostHealth.cost",
        "charmable.active",
        "grabbable.isGrabbed",
        "itemNoReturn.active",
        *(s + ".active" for s in STATUSES),
        *("marker." + m for m in MARKERS),
    }
)


def world_contract(capacity: int, profile: str) -> dict:
    if type(capacity) is not int or not 1 <= capacity <= 4096:
        raise ValueError("World capacity must be an integer in [1, 4096]")
    if profile not in {"player-v1", "internal-v1"}:
        raise ValueError("World observations require an explicit information profile")
    return {
        "version": 1,
        "capacity": capacity,
        "profile": profile,
        "registry_digest": world_registry()[1],
        "facts": list(FACTS),
        "entity_fields": ["catalog_id", "x", "y", "currency", "grab", "grabber"],
        "terrain": "currently-visible-exact-types-v1",
    }


def world_spaces(capacity: int) -> dict:
    world_contract(capacity, "internal-v1")
    return {
        "world_entities": spaces.Box(-(2**31), 2**31 - 1, (capacity, 6), np.int32),
        "world_facts": spaces.Box(-(2**31), 2**31 - 1, (capacity, len(FACTS), 3), np.int32),
        "world_tiles": spaces.Box(0, len(world_registry()[0]["tiles"]), (21, 21), np.int32),
        "world_current": spaces.Box(0, 1, (1,), np.int8),
    }


def _integer(value, low=-(2**31), high=2**31 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid world integer or out-of-range value")
    return value


def _mapping(value):
    if value == []:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected world component mapping")
    return value


def decode_world_snapshot(payload):
    if payload is None:
        return None
    if (
        not isinstance(payload, dict)
        or type(payload.get("version")) is not int
        or payload["version"] != 1
    ):
        raise ValueError("Unsupported world snapshot")
    if type(payload.get("current")) is not bool:
        raise ValueError("Missing world freshness")
    coverage = payload.get("coverage")
    if (
        not isinstance(coverage, list)
        or not all(isinstance(f, str) for f in coverage)
        or sorted(coverage) != sorted(FIELDS)
    ):
        raise ValueError("Incomplete world field coverage")
    if not isinstance(payload.get("entities"), list) or not isinstance(payload.get("tiles"), list):
        raise ValueError("World records require arrays")
    registry, digest = world_registry()
    entities, seen = [], set()
    for row in payload["entities"]:
        if not isinstance(row, dict):
            raise ValueError("Invalid world entity")
        identity = _integer(row.get("id"), 1, 2**53 - 1)
        if identity in seen:
            raise ValueError("Duplicate world entity ID")
        if not isinstance(row.get("name"), str) or row["name"] not in registry["entity_ids"]:
            raise ValueError("Unknown world identity: " + str(row.get("name")))
        seen.add(identity)
        x, y = (_integer(row.get(a), -10, 10) for a in ("x", "y"))
        if type(row.get("displayed")) is not bool:
            raise ValueError("Missing world display evidence")
        components = _mapping(row.get("components"))
        expected_components = {pair[0] for pair in registry["fields"]}
        if not set(components) <= expected_components:
            raise ValueError("Unknown world component")
        values = {}
        for component, field in registry["fields"]:
            key = component + "." + field
            if component not in components:
                values[key] = [0, 0, 0]
                continue
            value = _mapping(components[component]).get(field)
            if type(value) is bool:
                value = int(value)
            values[key] = [_integer(value), 1, 1]
        markers = row.get("markers")
        if (
            not isinstance(markers, list)
            or any(m not in MARKERS for m in markers)
            or len(set(markers)) != len(markers)
        ):
            raise ValueError("Invalid world markers")
        for marker in MARKERS:
            values["marker." + marker] = [int(marker in markers), 1, 1]
        for status in STATUSES:
            turns, permanent = values[status + ".remainingTurns"], values[status + ".permanent"]
            values[status + ".active"] = [int(turns[0] > 0 or permanent[0] != 0), 1, 1]
        currency = row.get("currency")
        if currency is not None and currency not in registry["currencies"]:
            raise ValueError("Unknown world price currency: " + str(currency))
        if ("priceTagCostCurrency" in components) != (currency is not None):
            raise ValueError("Incomplete world price")
        relations = _mapping(row.get("relations"))
        if not set(relations) <= {"grab", "grabber"}:
            raise ValueError("Unknown world relation")
        for target in relations.values():
            _integer(target, 1, 2**53 - 1)
        entities.append(
            {
                "id": identity,
                "name": row["name"],
                "catalog_id": registry["entity_ids"][row["name"]],
                "x": x,
                "y": y,
                "displayed": row["displayed"],
                "facts": values,
                "currency": registry["currencies"].index(currency) + 1 if currency else 0,
                "relations": relations,
            }
        )
    tiles, seen_tiles = [], set()
    for tile in payload["tiles"]:
        if (
            not isinstance(tile, dict)
            or not isinstance(tile.get("name"), str)
            or tile["name"] not in registry["tile_ids"]
        ):
            raise ValueError("Unknown world terrain identity")
        xy = tuple(_integer(tile.get(a), -10, 10) for a in ("x", "y"))
        if xy in seen_tiles:
            raise ValueError("Duplicate world tile")
        seen_tiles.add(xy)
        tiles.append((*xy, registry["tile_ids"][tile["name"]]))
    return {
        "current": payload["current"],
        "entities": entities,
        "tiles": tiles,
        "registry_digest": digest,
    }


def project_world(state: dict, capacity: int, profile: str) -> dict:
    contract = world_contract(capacity, profile)
    if state["registry_digest"] != contract["registry_digest"]:
        raise ValueError("World registry mismatch")
    result = {
        key: np.zeros(space.shape, dtype=space.dtype)
        for key, space in world_spaces(capacity).items()
    }
    if not state["current"]:
        return result
    rows = [r for r in state["entities"] if profile == "internal-v1" or r["displayed"]]
    if len(rows) > capacity:
        raise ValueError(f"World capacity {capacity} exceeded by {len(rows)} entities")
    indices = {r["id"]: i + 1 for i, r in enumerate(rows)}
    for i, row in enumerate(rows):
        result["world_entities"][i] = (
            row["catalog_id"],
            row["x"],
            row["y"],
            row["currency"],
            indices.get(row["relations"].get("grab"), 0),
            indices.get(row["relations"].get("grabber"), 0),
        )
        for j, field in enumerate(FACTS):
            result["world_facts"][i, j] = (
                row["facts"][field]
                if profile == "internal-v1" or field in PLAYER_FIELDS
                else [0, 0, 1]
            )
    for x, y, tile_id in state["tiles"]:
        result["world_tiles"][y + 10, x + 10] = tile_id
    result["world_current"][0] = 1
    return result
