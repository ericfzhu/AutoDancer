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
    target_terrain_after: int | None = None
    event_kinds: tuple[str, ...] = ()

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


def _target_cell(
    observation: dict[str, np.ndarray], action: Action
) -> tuple[int, int] | None:
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
    terrain_before = actor_before = terrain_after = None
    if target is not None:
        row, column = target
        terrain_before = int(before["grid"][row, column, GridChannel.TERRAIN_CLASS])
        actor_before = int(before["grid"][row, column, GridChannel.ACTOR_CLASS])
        # This comparison is valid only when Bard remained centred on the same
        # world coordinate. If Bard moved, the after-grid target is a new cell.
        if not position_changed and not floor_changed:
            terrain_after = int(after["grid"][row, column, GridChannel.TERRAIN_CLASS])

    if floor_changed:
        category = "floor_transition"
    elif events & COMBAT_EVENTS:
        category = "combat"
    elif events & INTERACTION_EVENTS:
        category = "interaction"
    elif action == Action.WAIT:
        category = "wait"
    elif target is not None and terrain_before == int(Terrain.WALL) and (
        position_changed or terrain_after != terrain_before
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
        target_terrain_after=terrain_after,
        event_kinds=event_kinds,
    )
