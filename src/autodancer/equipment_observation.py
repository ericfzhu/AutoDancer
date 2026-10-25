"""Explicit equipment information profiles; never enabled by the legacy profile.

Capacity is a configurable allocation budget, not a claim about the game's
inventory limit. Overflow rejects the observation instead of dropping items.
Entity IDs are used only to join relations, never emitted as policy features.
"""

from copy import deepcopy
from typing import Any

import numpy as np
from gymnasium import spaces

from autodancer.equipment_state import (
    CAPABILITY_COMPONENTS,
    EQUIPMENT_FIELDS,
    SPECIAL_ACTIONS,
    equipment_registry,
)

EQUIPMENT_PROFILES = ("legacy", "internal-v1", "player-v1")
EQUIPMENT_SLOTS = (
    "weapon",
    "shovel",
    "body",
    "head",
    "feet",
    "ring",
    "torch",
    "bomb",
    "action",
    "spell",
    "hud",
    "misc",
    "follower",
    "innate",
    "shrine",
    "stash",
)
EQUIPMENT_FACTS = (
    *("capability." + name for name in CAPABILITY_COMPONENTS),
    *(field.name for field in EQUIPMENT_FIELDS),
    "spell_usable",
    "ban.pickup",
    "ban.equip",
    "ban.activate",
)
# Identity, ownership, parent row (one-based, zero = none), conversion identity;
# then one position per slot and one ownership bit per special input.
SLOT_OFFSET = 5  # Column 4 preserves the position within a multi-item container.
CONTROL_OFFSET = SLOT_OFFSET + len(EQUIPMENT_SLOTS)
ITEM_FEATURES = CONTROL_OFFSET + len(SPECIAL_ACTIONS)


def equipment_contract(capacity: int, profile: str = "internal-v1") -> dict[str, Any]:
    if type(capacity) is not int or not 1 <= capacity <= 4096:
        raise ValueError("equipment capacity must be an integer in [1, 4096]")
    if profile not in EQUIPMENT_PROFILES[1:]:
        raise ValueError("Select an equipment information profile")
    from autodancer.equipment_state import PLAYER_EFFECT_NAMES

    return {
        "version": 3,
        "profile": profile,
        "capacity": capacity,
        "registry_digest": equipment_registry()[1],
        "slots": list(EQUIPMENT_SLOTS),
        "facts": list(EQUIPMENT_FACTS),
        "controls": [action.name for action in SPECIAL_ACTIONS],
        "player_effects": list(PLAYER_EFFECT_NAMES),
        "world_fields": ["kind", "catalog_id", "x", "y"],
    }


def equipment_spaces(capacity: int) -> dict[str, spaces.Box]:
    contract = equipment_contract(capacity)
    return {
        "equipment_items": spaces.Box(
            0, 2**31 - 1, shape=(capacity, ITEM_FEATURES), dtype=np.int32
        ),
        # Last axis = value, known, applicable. Int32 preserves large quantities.
        "equipment_facts": spaces.Box(
            -(2**31), 2**31 - 1, shape=(capacity, len(EQUIPMENT_FACTS), 3), dtype=np.int32
        ),
        "equipment_current": spaces.Box(0, 1, shape=(1,), dtype=np.int8),
        "equipment_effects": spaces.Box(
            -(2**31), 2**31 - 1, shape=(len(contract["player_effects"]), 3), dtype=np.int32
        ),
        "equipment_world": spaces.Box(-(2**31), 2**31 - 1, shape=(capacity, 4), dtype=np.int32),
        "equipment_world_current": spaces.Box(0, 1, shape=(1,), dtype=np.int8),
    }


def equipment_environment_options(architecture: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the equipment environment required by an A9 checkpoint."""
    if architecture.get("version") != 9:
        return {}
    contract = architecture.get("equipment", {})
    if contract != equipment_contract(contract.get("capacity"), contract.get("profile")):
        raise ValueError("Equipment checkpoint contract or registry does not match")
    return {
        "input_mode": "engine-input-v1",
        "equipment_profile": contract["profile"],
        "equipment_capacity": contract["capacity"],
    }


def carried_item_counts(observation: dict[str, np.ndarray]) -> dict[int, int]:
    """Count owned items and stored contents, excluding virtual action owners.

    This is inventory-change evidence, not proof of a world pickup: conversions
    and grants can also change these counts.
    """
    items, facts = observation["equipment_items"], observation["equipment_facts"]
    if not observation["equipment_current"][0]:
        return {}
    carried = (items[:, 0] != 0) & (items[:, 1] != 0)
    parents = items[:, 2]
    for _ in range(len(items)):
        expanded = carried | ((parents > 0) & np.r_[False, carried][parents])
        if np.array_equal(expanded, carried):
            break
        carried = expanded
    result: dict[int, int] = {}
    for index in np.flatnonzero(carried):
        quantity, known, applicable = facts[index, EQUIPMENT_FACTS.index("quantity")]
        if applicable and not known:
            raise ValueError("Cannot infer inventory counts from unknown stack quantity")
        count = int(quantity) if applicable else 1
        identity = int(items[index, 0])
        result[identity] = result.get(identity, 0) + count
    return result


def project_equipment(
    state: dict[str, Any], capacity: int, profile: str = "internal-v1"
) -> dict[str, np.ndarray]:
    """Encode a validated snapshot under the selected information profile."""
    contract = equipment_contract(capacity, profile)
    if (
        state.get("version") not in {2, 3}
        or state.get("registry_digest") != contract["registry_digest"]
    ):
        raise ValueError("Equipment observation requires the pinned registry")
    result = {
        "equipment_items": np.zeros((capacity, ITEM_FEATURES), dtype=np.int32),
        "equipment_facts": np.zeros((capacity, len(EQUIPMENT_FACTS), 3), dtype=np.int32),
        "equipment_current": np.zeros(1, dtype=np.int8),
        "equipment_effects": np.zeros((len(contract["player_effects"]), 3), dtype=np.int32),
        "equipment_world": np.zeros((capacity, 4), dtype=np.int32),
        "equipment_world_current": np.zeros(1, dtype=np.int8),
    }
    # Terminal snapshots can contain the last running inventory. Do not present
    # that historical payload as current equipment to either actor or critic.
    if not state["current"]:
        return result
    if profile == "player-v1":
        state = player_equipment_view(state)
    items = state["items"]
    if len(items) > capacity:
        raise ValueError(f"Equipment capacity {capacity} exceeded by {len(items)} items")
    rows = {item["id"]: index for index, item in enumerate(items)}
    owned = set(state["owned"])
    registry = equipment_registry()[0]
    world = state.get("world") or []
    result["equipment_world_current"][0] = state.get("world") is not None
    if len(world) > capacity:
        raise ValueError(f"Visible equipment capacity {capacity} exceeded by {len(world)} objects")
    for index, row in enumerate(world):
        result["equipment_world"][index] = (
            {"item": 1, "charmed": 2, "no_return": 3}[row["kind"]],
            registry[row["name"]] if row["kind"] in {"item", "no_return"} else 0,
            row["x"],
            row["y"],
        )
    values, facts = result["equipment_items"], result["equipment_facts"]

    def checked(value: int) -> int:
        if not -(2**31) <= value < 2**31:
            raise ValueError("Equipment value exceeds int32; cannot encode losslessly")
        return value

    for index, item in enumerate(items):
        target = item.get("conversion_target")
        if target and target not in registry:
            raise ValueError("Unknown equipment conversion target: " + target)
        values[index, :2] = item["catalog_id"], item["id"] in owned
        values[index, 3] = registry[target] if target else 0
        for column, name in enumerate(EQUIPMENT_FACTS):
            fact = item["facts"][name]
            facts[index, column] = (
                checked(int(fact["value"])) if fact["known"] else 0,
                fact["known"],
                fact["applicable"],
            )
    for slot in state["slots"]:
        if slot["name"] not in EQUIPMENT_SLOTS:
            raise ValueError("Unknown equipment slot: " + slot["name"])
        column = SLOT_OFFSET + EQUIPMENT_SLOTS.index(slot["name"])
        row = rows[slot["item_id"]]
        if values[row, column]:
            raise ValueError("Item occupies multiple positions in one slot")
        values[row, column] = checked(slot["index"])
    for relation in state["containers"]:
        if relation["content_id"] is not None:
            values[rows[relation["content_id"]], 2] = rows[relation["item_id"]] + 1
            values[rows[relation["content_id"]], 4] = relation.get("index", 1)
    for control in state["controls"]:
        if control["owner_id"] is not None:
            column = CONTROL_OFFSET + contract["controls"].index(control["action"])
            values[rows[control["owner_id"]], column] = 1
    result["equipment_current"][0] = 1
    for index, name in enumerate(contract["player_effects"]):
        fact = state["player_effects"][name]
        result["equipment_effects"][index] = (
            checked(int(fact["value"])) if fact["known"] else 0,
            fact["known"],
            fact["applicable"],
        )
    return result


def player_equipment_view(state: dict[str, Any]) -> dict[str, Any]:
    """Expose current HUD identities/counters; keep unobserved facts unknown.

    No engine-selected owner, hidden storage state or applicability bit is used
    to fill gaps in the display. A recurrent policy can remember prior displays.
    """
    if state.get("display") is None:
        raise ValueError("player-v1 requires authoritative equipment display evidence")
    view = deepcopy(state)
    shown = {r["id"]: set(r["fields"]) for r in state["display"]["items"]}
    view["items"] = [i for i in view["items"] if i["id"] in shown]
    unknown = dict(value=None, known=False, applicable=True, source="visible")
    for item in view["items"]:
        item.pop("conversion_target", None)
        item["facts"] = {
            name: {**fact, "source": "visible"} if name in shown[item["id"]] else dict(unknown)
            for name, fact in item["facts"].items()
        }
    view["owned"] = [i for i in view["owned"] if i in shown]
    view["slots"] = [s for s in view["slots"] if s["item_id"] in shown]
    view["containers"] = []  # Stored identities can be displayed without their hidden internals.
    view["controls"] = [
        {**r, "owner_id": r["owner_id"] or None} for r in state["display"]["controls"]
    ]
    view["player_effects"] = {
        name: {**fact, "source": "visible"}
        if name in {"cursed_health", "groove_multiplier"}
        or name.endswith("_active")
        and name != "risk_active"
        else dict(unknown)
        for name, fact in view["player_effects"].items()
    }
    return view
