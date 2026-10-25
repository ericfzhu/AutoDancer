"""Map inspected Lua component fields into the next harness's state contract.

The live extension supplies engine-selected owners and current item fields.
Ownership and bans do not establish executable availability or loot reachability.
Sources are conservatively INTERNAL until visibility is verified.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from autodancer.constants import Action
from autodancer.harness_contract import Availability, ControlState, InformationSource, StateFact


@dataclass(frozen=True)
class EquipmentField:
    name: str
    component: str
    field: str
    boolean: bool = False


EQUIPMENT_FIELDS = (
    EquipmentField("activation_active", "itemActivable", "active", True),
    EquipmentField("toggle_active", "itemToggleable", "active", True),
    EquipmentField("ammo", "weaponReloadable", "ammo"),
    EquipmentField("ammo_capacity", "weaponReloadable", "maximumAmmo"),
    EquipmentField("ammo_per_reload", "weaponReloadable", "ammoPerReload"),
    EquipmentField("quantity", "itemStack", "quantity"),
    EquipmentField("weapon_component_damage", "weapon", "damage"),
    EquipmentField("cooldown_turns", "spellCooldownTime", "remainingTurns"),
    EquipmentField("cooldown_kills", "spellCooldownKills", "remainingKills"),
    EquipmentField("combo", "itemComboable", "combo"),
    EquipmentField("activation_damage", "itemDamageOnActivation", "damage"),
    EquipmentField("blood_cast_damage", "spellBloodMagic", "damage"),
    EquipmentField("bag_action_capacity", "itemBag", "actionSlotCapacity"),
)
V2_FIELDS = EQUIPMENT_FIELDS
EQUIPMENT_FIELDS += (
    EquipmentField("equipped", "item", "equipped", True),
    EquipmentField("damage_boost_turns", "itemConditionalDamageIncrease", "remainingTurns"),
    EquipmentField("conditional_armor", "itemArmorGrooveChain", "damageReduction"),
    EquipmentField("invincible", "itemConditionalInvincibility", "active", True),
    EquipmentField("invincibility_on_kill", "itemConditionalInvincibilityOnKill", "active", True),
    EquipmentField("dig_boost_active", "itemConditionalDigStrengthIncrease", "active", True),
    EquipmentField("dig_boost_pending", "itemConditionalDigStrengthOnActivation", "pending", True),
    EquipmentField("dash_on_kill", "itemDashOnKill", "active", True),
    EquipmentField("dash_on_dig", "itemDashOnDig", "active", True),
    EquipmentField("pickup_health_gained", "itemIncreaseMaxHealth", "healthGained"),
    EquipmentField("first_pickup_heal_used", "itemHealOnFirstPickup", "done", True),
    EquipmentField("pickup_damage_active", "itemDamageOnPickup", "active", True),
    EquipmentField("parry_active", "weaponParryOnAttack", "active", True),
    EquipmentField("combo_reset", "itemComboable", "reset", True),
    EquipmentField("holster_sticky", "itemHolster", "sticky", True),
    EquipmentField("holster_just_swapped", "itemHolsterAutoSwap", "justSwapped", True),
    EquipmentField("bonus_item_granted", "itemGrantBonusItem", "granted", True),
    EquipmentField("stash_health", "health", "health"),
    EquipmentField("stash_max_health", "health", "maxHealth"),
    EquipmentField("stash_cursed_health", "cursedHealth", "health"),
    EquipmentField("no_return_active", "itemNoReturn", "active", True),
    EquipmentField("stack_dupable", "itemStackDupable", "active", True),
    EquipmentField("overlay_active", "itemOverlaySpriteHUD", "active", True),
    EquipmentField("tunnel_active", "Coldsteel_itemTunnelAttack", "active", True),
    EquipmentField("leek_combo_active", "Coldsteel_weaponComboTracker", "active", True),
    EquipmentField("leek_combo", "Coldsteel_weaponComboTracker", "combo"),
    EquipmentField("leek_combo_pending", "Coldsteel_weaponComboable", "pendingIncrease", True),
    EquipmentField("pogo_collectible", "Goldman_itemCollectibleDuringPogo", "active", True),
)

CAPABILITY_COMPONENTS = {
    "weapon": "weapon",
    "throwable": "weaponThrowable",
    "reloadable": "weaponReloadable",
    "activable": "itemActivable",
    "toggleable": "itemToggleable",
    "holster": "itemHolster",
    "bag": "itemBag",
    "spell_cast": "itemCastOnUse",
    "convertible": "itemConvertible",
}
V2_COMPONENTS = frozenset(CAPABILITY_COMPONENTS.values()) | frozenset(
    field.component for field in V2_FIELDS
)
CAPABILITY_COMPONENTS["storage"] = "storage"

PLAYER_EFFECT_FIELDS = (
    *(
        EquipmentField(f"{component}_turns", component, "remainingTurns")
        for component in (
            "invincibility",
            "barrier",
            "gigantism",
            "dwarfism",
            "freezable",
            "confusable",
        )
    ),
    *(
        EquipmentField(f"{component}_permanent", component, "permanent", True)
        for component in (
            "invincibility",
            "barrier",
            "gigantism",
            "dwarfism",
            "freezable",
            "confusable",
        )
    ),
    EquipmentField("regeneration_kills", "regenerationKillCounter", "killCount"),
    EquipmentField("regeneration_threshold", "regenerationKillCounter", "killThreshold"),
    EquipmentField("regeneration_increment", "regenerationKillCounter", "totalIncrement"),
    EquipmentField("cursed_health", "cursedHealth", "health"),
    EquipmentField("groove_kills", "grooveChain", "killCount"),
    EquipmentField("groove_multiplier", "grooveChain", "multiplier"),
    EquipmentField("risk_active", "riskDamage", "active", True),
    EquipmentField("risk_pending", "riskDamage", "pending", True),
    EquipmentField("charmed", "charmable", "active", True),
)
DISPLAY_FIELDS = frozenset(
    {
        "equipped",
        "ammo",
        "combo",
        "quantity",
        "activation_active",
        "toggle_active",
        "cooldown_turns",
        "cooldown_kills",
    }
)
VISIBLE_STATUS_COMPONENTS = (
    "invincibility",
    "barrier",
    "gigantism",
    "dwarfism",
    "freezable",
    "confusable",
)
PLAYER_EFFECT_NAMES = tuple(f.name for f in PLAYER_EFFECT_FIELDS) + tuple(
    c + "_active" for c in VISIBLE_STATUS_COMPONENTS
)


def decode_player_effects(payload):
    """Equipment can leave state on its holder after the item is consumed."""
    if payload == []:
        payload = {}
    if payload is not None and not isinstance(payload, Mapping):
        raise ValueError("Invalid player equipment effects")
    result = {}
    for binding in PLAYER_EFFECT_FIELDS:
        component = (payload or {}).get(binding.component, {})
        if component == []:
            component = {}
        if not isinstance(component, Mapping):
            raise ValueError("Invalid player effect component")
        value = component.get(binding.field)
        if value is not None and type(value) is not (bool if binding.boolean else int):
            raise ValueError("Invalid player effect field: " + binding.name)
        result[binding.name] = asdict(
            StateFact(
                value,
                value is not None,
                payload is None or binding.component in payload,
                InformationSource.INTERNAL,
            )
        )
    for component in VISIBLE_STATUS_COMPONENTS:
        turns, permanent = result[component + "_turns"], result[component + "_permanent"]
        known = turns["known"] and permanent["known"]
        value = (turns["value"] > 0 or permanent["value"]) if known else None
        result[component + "_active"] = asdict(
            StateFact(value, known, turns["applicable"], InformationSource.VISIBLE)
        )
    return result


def map_equipment_state(
    components: Mapping[str, Any], *, complete: bool = False
) -> dict[str, StateFact]:
    """Project a single runtime item snapshot, preserving unknownness.

    `complete` certifies component PRESENCE coverage for all names used here,
    not field completeness. An absent component in a partial snapshot is unknown.
    An empty component exists; missing fields remain unknown, never zero-filled.
    Lua empty tables may arrive as [] through the existing probe JSON encoder.
    """
    if not isinstance(components, Mapping) or type(complete) is not bool:
        raise ValueError("Expected component mapping and boolean completeness flag")
    result = {}
    source = InformationSource.INTERNAL
    for name, component in CAPABILITY_COMPONENTS.items():
        present = component in components
        if present and not isinstance(components[component], Mapping):
            marker = components[component]
            if not isinstance(marker, list) or marker:
                raise ValueError(f"Malformed capability component {component}")
        known = present or complete
        result[f"capability.{name}"] = StateFact(present if known else None, known, True, source)
    for binding in EQUIPMENT_FIELDS:
        if binding.component not in components:
            result[binding.name] = StateFact(None, False, not complete, source)
            continue
        values = components[binding.component]
        if isinstance(values, list) and not values:
            values = {}
        if not isinstance(values, Mapping):
            raise ValueError(f"Malformed component {binding.component}")
        if binding.field not in values:
            result[binding.name] = StateFact(None, False, True, source)
            continue
        value = values[binding.field]
        expected_type = bool if binding.boolean else int
        if type(value) is not expected_type:
            raise ValueError(
                f"{binding.component}.{binding.field} requires {expected_type.__name__}"
            )
        result[binding.name] = StateFact(value, True, True, source)
    return result


EQUIPMENT_COMPONENTS = frozenset(CAPABILITY_COMPONENTS.values()) | frozenset(
    field.component for field in EQUIPMENT_FIELDS
)
SPECIAL_ACTIONS = tuple(action for action in Action if action >= Action.BOMB)
V1_COMPONENTS = V2_COMPONENTS - {
    "itemComboable",
    "itemDamageOnActivation",
    "spellBloodMagic",
    "itemCastOnUse",
    "itemConvertible",
}


@cache
def equipment_registry() -> tuple[Mapping[str, int], str]:
    """Stable, persisted identities; inclusion does not grant mechanic support."""
    data = json.loads(files("autodancer").joinpath("data/equipment_registry.json").read_text())
    rows = data["items"]
    names = [row["name"] for row in rows]
    ids = [row["catalog_id"] for row in rows]
    if (
        data["version"] != 1
        or len(set(names)) != len(names)
        or ids != list(range(1, len(rows) + 1))
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("Invalid persisted equipment registry")
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MappingProxyType(dict(zip(names, ids, strict=True))), digest


def decode_equipment_snapshot(payload: Any) -> dict[str, Any] | None:
    """Validate additive live telemetry for info, never for actor tensors/masks.

    Missing extensions remain distinguishable from an empty inventory. Entity IDs
    are diagnostic join keys; field facts retain their information provenance.
    Terminal snapshots may be retained from the previous running state, in which
    case `current` must be false. No owner (including no item) proves availability.
    """
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("Equipment snapshot must be an object")
    if type(payload.get("version")) is not int or payload["version"] not in {1, 2, 3}:
        raise ValueError("Unsupported equipment snapshot version")
    version = payload["version"]
    registry, registry_digest = equipment_registry() if version >= 2 else ({}, None)
    if type(payload.get("current")) is not bool:
        raise ValueError("Equipment snapshot needs a boolean current flag")
    coverage = payload.get("coverage")
    expected_coverage = {1: V1_COMPONENTS, 2: V2_COMPONENTS, 3: EQUIPMENT_COMPONENTS}[version]
    if (
        not isinstance(coverage, list)
        or any(not isinstance(name, str) for name in coverage)
        or len(coverage) != len(expected_coverage)
        or set(coverage) != expected_coverage
    ):
        raise ValueError("Incomplete equipment component coverage")
    for name in ("items", "slots", "controls"):
        if not isinstance(payload.get(name), list):
            raise ValueError(f"Equipment {name} must be an array")

    def integer(value: Any, *, minimum: int = 1) -> int:
        if type(value) is not int or value < minimum:
            raise ValueError("Invalid equipment ID/index")
        return value

    items = {}
    for item in payload["items"]:
        if not isinstance(item, Mapping):
            raise ValueError("Equipment item must be an object")
        entity_id = integer(item.get("id"))
        if entity_id in items or not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError("Duplicate or unnamed equipment item")
        components = item.get("components")
        if components == []:  # Lua encodes empty tables as arrays.
            components = {}
        if not isinstance(components, Mapping) or not set(components) <= EQUIPMENT_COMPONENTS:
            raise ValueError("Unexpected equipment components")
        facts = map_equipment_state(components, complete=True)
        for binding in EQUIPMENT_FIELDS:
            if binding.component not in expected_coverage or (
                version < 3 and binding not in V2_FIELDS
            ):
                facts[binding.name] = StateFact(None, False, True, InformationSource.INTERNAL)
        for capability, component in CAPABILITY_COMPONENTS.items():
            if component not in expected_coverage:
                facts["capability." + capability] = StateFact(
                    None, False, True, InformationSource.INTERNAL
                )
        if version >= 2 and "itemCastOnUse" in components:
            if type(item.get("spell_usable")) is not bool:
                raise ValueError("Missing engine spell usability")
            facts["spell_usable"] = StateFact(
                item["spell_usable"], True, True, InformationSource.INTERNAL
            )
        else:
            facts["spell_usable"] = StateFact(None, False, version == 1, InformationSource.INTERNAL)
        bans = item.get("bans")
        if (
            not isinstance(bans, Mapping)
            or set(bans) != {"pickup", "equip", "activate"}
            or any(type(value) is not bool for value in bans.values())
        ):
            raise ValueError("Incomplete equipment use bans")
        facts.update(
            ("ban." + phase, StateFact(value, True, True, InformationSource.INTERNAL))
            for phase, value in bans.items()
        )
        items[entity_id] = {
            "id": entity_id,
            "name": item["name"],
            "facts": {name: asdict(fact) for name, fact in facts.items()},
        }
        if version >= 2:
            if item["name"] not in registry:
                raise ValueError("Equipment type is outside the pinned registry: " + item["name"])
            items[entity_id]["catalog_id"] = registry[item["name"]]
        if "conversion_target" in item:
            if (
                not isinstance(item["conversion_target"], str)
                or "itemConvertible" not in components
            ):
                raise ValueError("Invalid equipment conversion target")
            items[entity_id]["conversion_target"] = item["conversion_target"]

    slots = []
    seen_slots = set()
    for slot in payload["slots"]:
        if (
            not isinstance(slot, Mapping)
            or not isinstance(slot.get("name"), str)
            or not slot["name"]
        ):
            raise ValueError("Invalid equipment slot")
        index, entity_id = integer(slot.get("index")), integer(slot.get("item_id"))
        key = (slot["name"], index)
        if key in seen_slots or entity_id not in items:
            raise ValueError("Duplicate slot or missing equipment item")
        seen_slots.add(key)
        slots.append({"name": slot["name"], "index": index, "item_id": entity_id})

    controls = {}
    for row in payload["controls"]:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("action"), str)
            or row["action"] not in {a.name for a in SPECIAL_ACTIONS}
        ):
            raise ValueError("Unknown equipment action")
        action = Action[row["action"]]
        owner_id = integer(row.get("owner_id"), minimum=0)
        if action in controls or (owner_id and owner_id not in items):
            raise ValueError("Duplicate action or missing owner item")
        control = ControlState(
            action, Availability.UNKNOWN, owner=items[owner_id]["name"] if owner_id else None
        )
        controls[action] = {
            **asdict(control),
            "action": action.name,
            "owner_id": owner_id or None,
            "selection_known": payload["current"],
        }
    if (payload["current"] or controls) and set(controls) != set(SPECIAL_ACTIONS):
        raise ValueError("Incomplete equipment owner selection")
    containers = []
    owned = None
    if version >= 2:
        owned = payload.get("owned")
        if not isinstance(owned, list):
            raise ValueError("Missing owned-item inventory")
        owned = [integer(value) for value in owned]
        if len(set(owned)) != len(owned) or not set(owned) <= items.keys():
            raise ValueError("Duplicate or missing owned item")
        if not isinstance(payload.get("containers"), list):
            raise ValueError("Missing container state")
        parents = {}
        seen_containers = set()
        for row in payload["containers"]:
            if not isinstance(row, Mapping):
                raise ValueError("Invalid container relation")
            parent = integer(row.get("item_id"))
            child = integer(row.get("content_id"), minimum=0)
            kind = row.get("kind", "holster")
            index = integer(row.get("index", 1))
            key = (parent, kind, index)
            if parent not in items or key in seen_containers:
                raise ValueError("Missing or duplicate container")
            if kind not in ({"holster", "storage"} if version == 3 else {"holster"}):
                raise ValueError("Invalid container kind")
            if items[parent]["facts"]["capability." + kind]["value"] is not True:
                raise ValueError("Container lacks " + kind + " capability")
            if kind == "holster" and index != 1:
                raise ValueError("Holsters contain one item")
            seen_containers.add(key)
            if child:
                if child not in items or child in parents or child == parent:
                    raise ValueError("Invalid or multiply contained item")
                parents[child] = parent
            relation = {"item_id": parent, "content_id": child or None}
            if version == 3:
                relation.update(kind=kind, index=index)
            containers.append(relation)
        expected_containers = {
            (i, kind)
            for i, item in items.items()
            for kind in ("holster", "storage")
            if item["facts"]["capability." + kind]["value"]
        }
        if {(p, k) for p, k, _ in seen_containers} != expected_containers:
            raise ValueError("Incomplete holster contents")
        for parent, kind in expected_containers:
            group = [
                r for r in containers if r["item_id"] == parent and r.get("kind", "holster") == kind
            ]
            indices = sorted(r.get("index", 1) for r in group)
            if indices != list(range(1, len(group) + 1)):
                raise ValueError("Noncontiguous container contents")
            if len(group) > 1 and any(r["content_id"] is None for r in group):
                raise ValueError("Empty container marker mixed with contents")
        for child in parents:
            seen = {child}
            current = child
            while current in parents:
                current = parents[current]
                if current in seen:
                    raise ValueError("Containment cycle")
                seen.add(current)
        if set(parents) & {slot["item_id"] for slot in slots}:
            raise ValueError("Stored item also occupies an inventory slot")
    world = payload.get("world")
    if world is not None:
        if not isinstance(world, list):
            raise ValueError("Invalid equipment world evidence")
        for row in world:
            if not isinstance(row, Mapping) or row.get("kind") not in {
                "item",
                "charmed",
                "no_return",
            }:
                raise ValueError("Invalid visible equipment object")
            if any(
                type(row.get(axis)) is not int or not -10 <= row[axis] <= 10 for axis in ("x", "y")
            ):
                raise ValueError("Visible equipment outside observation grid")
            if row["kind"] in {"item", "no_return"} and row.get("name") not in registry:
                raise ValueError("Visible equipment outside pinned registry")
    display = payload.get("display")
    if display is not None:
        if not isinstance(display, Mapping) or set(display) != {"items", "controls"}:
            raise ValueError("Invalid equipment display")
        if not isinstance(display["items"], list) or not isinstance(display["controls"], list):
            raise ValueError("Equipment display requires arrays")
        shown = set()
        for row in display["items"]:
            if not isinstance(row, Mapping):
                raise ValueError("Invalid displayed item")
            identity = integer(row.get("id"))
            fields = row.get("fields")
            if identity not in items or identity in shown or not isinstance(fields, list):
                raise ValueError("Missing or duplicate displayed item")
            if (
                any(not isinstance(f, str) for f in fields)
                or not set(fields) <= DISPLAY_FIELDS
                or len(set(fields)) != len(fields)
            ):
                raise ValueError("Unsupported displayed equipment field")
            shown.add(identity)
        actions = set()
        for row in display["controls"]:
            if not isinstance(row, Mapping) or row.get("action") not in {
                a.name for a in SPECIAL_ACTIONS
            }:
                raise ValueError("Invalid displayed control")
            owner = integer(row.get("owner_id"), minimum=0)
            if row["action"] in actions or (owner and owner not in shown):
                raise ValueError("Missing or duplicate displayed owner")
            actions.add(row["action"])
        if payload["current"] and actions != {a.name for a in SPECIAL_ACTIONS}:
            raise ValueError("Incomplete displayed controls")
    return {
        "version": version,
        "registry_digest": registry_digest,
        "current": payload["current"],
        "items": list(items.values()),
        "slots": slots,
        "controls": [controls[a] for a in SPECIAL_ACTIONS if a in controls],
        "owned": owned,
        "containers": containers,
        "player_effects": decode_player_effects(payload.get("player_effects")),
        "display": display,
        "world": world,
    }
