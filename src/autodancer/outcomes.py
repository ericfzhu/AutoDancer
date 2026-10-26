"""Mechanic-level classification of authoritative live action outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from autodancer.constants import (
    DIRECTION_DELTAS,
    GRID_SIZE,
    Action,
    ActorKind,
    GridChannel,
    PlayerFeature,
    Terrain,
    TrapKind,
)

COMBAT_EVENTS = frozenset({"enemy_damage", "enemy_kill"})
INTERACTION_EVENTS = frozenset({"item_collected", "container_opened", "currency_collected"})


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    category: str
    position_changed: bool
    floor_changed: bool
    productive: bool
    target_terrain_before: int | None = None
    target_actor_before: int | None = None
    target_trap_before: int | None = None
    target_terrain_after: int | None = None
    descent_source: str | None = None
    event_kinds: tuple[str, ...] = ()
    equipment_action: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _position(observation: dict[str, np.ndarray]) -> tuple[int, int, int, int]:
    player = observation["player"]
    return (
        int(player[PlayerFeature.ZONE]),
        int(player[PlayerFeature.FLOOR]),
        int(player[PlayerFeature.X]),
        int(player[PlayerFeature.Y]),
    )


def _target_cell(observation: dict[str, np.ndarray], action: Action) -> tuple[int, int] | None:
    delta = DIRECTION_DELTAS.get(action)
    if delta is None:
        return None
    centre = GRID_SIZE // 2
    dx, dy = delta
    return centre + dy, centre + dx


def classify_action_outcome(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    action: Action | int,
    info: dict[str, Any],
) -> ActionOutcome:
    """Classify an action without pretending ambiguous engine outcomes are certain."""
    action = Action(int(action))
    before_position = _position(before)
    after_position = _position(after)
    floor_changed = before_position[:2] != after_position[:2]
    position_changed = before_position[2:] != after_position[2:]
    event_kinds = tuple(
        sorted({str(event.get("kind", "")) for event in info.get("raw_events", []) if event})
    )
    events = frozenset(event_kinds)
    target = _target_cell(before, action)
    terrain_before = actor_before = trap_before = terrain_after = None
    if target is not None:
        row, column = target
        terrain_before = int(before["grid"][row, column, GridChannel.TERRAIN_CLASS])
        actor_before = int(before["grid"][row, column, GridChannel.ACTOR_CLASS])
        trap_before = int(before["grid"][row, column, GridChannel.TRAP])
        if "world_entities" in before:
            from autodancer.world_observation import FACTS, world_registry

            names = world_registry()[0]["entities"]
            records = before["world_entities"]
            matches = np.flatnonzero(
                (records[:, 0] != 0) & (records[:, 1] == column - 10) & (records[:, 2] == row - 10)
            )
            actor_before = trap_before = 0
            for i in matches:
                if before["world_facts"][i, FACTS.index("marker.enemy"), 0]:
                    actor_before = int(ActorKind.OTHER)
                if names[int(records[i, 0]) - 1].startswith("Trapdoor"):
                    trap_before = int(TrapKind.TRAPDOOR)
        # This comparison is valid only when Bard remained centred on the same
        # world coordinate. If Bard moved, the after-grid target is a new cell.
        if not position_changed and not floor_changed:
            terrain_after = int(after["grid"][row, column, GridChannel.TERRAIN_CLASS])

    descent_source = None
    equipment_action = None
    before_controls = before.get("equipment_controls", (0, 0, 0))
    after_controls = after.get("equipment_controls", (0, 0, 0))
    if (
        action == Action.BOMB
        and before.get("bounded_loadout", (0, 0))[1]
        and not floor_changed
        and before["inventory"][6, 2] > after["inventory"][6, 2]
        and np.any(after["grid"][..., GridChannel.STATUS] & 1)
    ):
        equipment_action = "bomb_placed"
    if before_controls[0] == after_controls[0] == 1 and not floor_changed:
        same_weapon = np.array_equal(before["inventory"][0, :2], after["inventory"][0, :2])
        if before_controls[1] and after_controls[1] and same_weapon and action == Action.THROW:
            if not before_controls[2] and after_controls[2]:
                equipment_action = "throw_prepared"
            elif before_controls[2] and after_controls[2]:
                equipment_action = "throw_already_armed"
        elif before_controls[1] and before_controls[2] and target is not None:
            if not position_changed and after["inventory"][0, 0] == 0:
                equipment_action = "weapon_thrown"
    if floor_changed:
        category = "floor_transition"
        if trap_before == int(TrapKind.TRAPDOOR):
            descent_source = "trapdoor"
        elif terrain_before == int(Terrain.STAIRS):
            descent_source = "stairs"
        else:
            descent_source = "unknown"
    elif events & COMBAT_EVENTS:
        category = "combat"
    elif events & INTERACTION_EVENTS:
        category = "interaction"
    elif equipment_action is not None:
        category = equipment_action
    elif action == Action.WAIT:
        category = "wait"
    elif (
        target is not None
        and terrain_before == int(Terrain.WALL)
        and (position_changed or terrain_after != terrain_before)
    ):
        category = "dig"
    elif position_changed:
        category = "move"
    elif target is not None and actor_before > int(ActorKind.PLAYER):
        category = "combat_attempt"
    elif target is not None and terrain_before == int(Terrain.WALL):
        category = "wall_attempt"
    elif target is not None:
        category = "unchanged_direction"
    elif "equipment_items" in before and "equipment_items" in after:
        # A changed cooldown, toggle, container, holder status or visible target
        # is an observed transition, not proof that a particular effect succeeded.
        changed = any(
            not np.array_equal(before[key], after[key])
            for key in (
                "equipment_items",
                "equipment_facts",
                "equipment_effects",
                "equipment_world",
            )
        ) or not np.array_equal(before["player"][:4], after["player"][:4])
        if "world_entities" in before:
            changed = changed or any(
                not np.array_equal(before[k], after[k])
                for k in ("world_entities", "world_facts", "world_tiles")
            )
        category = "special_state_changed" if changed else "special_no_observed_change"
    else:
        category = "special_no_effect"
    productive = category in {"floor_transition", "combat", "interaction", "dig", "move"}
    return ActionOutcome(
        category=category,
        position_changed=position_changed,
        floor_changed=floor_changed,
        productive=productive,
        target_terrain_before=terrain_before,
        target_actor_before=actor_before,
        target_trap_before=trap_before,
        target_terrain_after=terrain_after,
        descent_source=descent_source,
        event_kinds=event_kinds,
        equipment_action=equipment_action,
    )
