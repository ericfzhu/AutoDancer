"""Mutable simulator state and raw turn events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autodancer.constants import ActorKind, ItemKind, Terrain, TrapKind


@dataclass(slots=True)
class Actor:
    entity_id: int
    kind: ActorKind
    x: int
    y: int
    health: int
    max_health: int
    damage: int = 1
    move_period: int = 1
    boss: bool = False

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(slots=True)
class GroundItem:
    kind: ItemKind
    x: int
    y: int
    value: int = 1


@dataclass(slots=True)
class Bomb:
    x: int
    y: int
    fuse: int = 3


@dataclass(slots=True, frozen=True)
class GameEvent:
    kind: str
    amount: int = 0
    entity_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "amount": self.amount}
        if self.entity_id is not None:
            result["entity_id"] = self.entity_id
        if self.data:
            result["data"] = self.data
        return result


@dataclass(slots=True)
class WorldState:
    width: int
    height: int
    terrain: np.ndarray
    traps: np.ndarray
    items: dict[tuple[int, int], GroundItem]
    enemies: dict[int, Actor]
    player: Actor
    stairs: tuple[int, int]
    visible: np.ndarray
    explored: np.ndarray
    zone: int = 1
    floor: int = 1
    turn: int = 0
    gold: int = 0
    groove: int = 0
    bombs: int = 1
    weapon_damage: int = 1
    facing: int = 1
    inventory: np.ndarray = field(
        default_factory=lambda: np.zeros((8, 3), dtype=np.int16)
    )
    active_bombs: list[Bomb] = field(default_factory=list)
    won: bool = False
    dead: bool = False

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def actor_at(self, x: int, y: int) -> Actor | None:
        if self.player.position == (x, y):
            return self.player
        for enemy in self.enemies.values():
            if enemy.position == (x, y):
                return enemy
        return None

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.terrain[y, x] in (
            Terrain.FLOOR,
            Terrain.STAIRS,
        )
