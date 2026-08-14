"""Clean-room level generation for curriculum tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autodancer.constants import ActorKind, ItemKind, Terrain, TrapKind
from autodancer.model import Actor, GroundItem, WorldState
from autodancer.rng import RandomChannels
from autodancer.tasks import TaskSpec


@dataclass(frozen=True, slots=True)
class EnemyDefinition:
    kind: ActorKind
    health: int
    damage: int
    move_period: int


ENEMY_REGISTRY: dict[int, tuple[EnemyDefinition, ...]] = {
    1: (
        EnemyDefinition(ActorKind.GREEN_SLIME, 1, 1, 2),
        EnemyDefinition(ActorKind.BLUE_SLIME, 2, 2, 2),
        EnemyDefinition(ActorKind.ZOMBIE, 1, 2, 2),
        EnemyDefinition(ActorKind.SKELETON, 1, 1, 2),
        EnemyDefinition(ActorKind.BAT, 1, 1, 1),
    ),
    2: (
        EnemyDefinition(ActorKind.BLUE_SLIME, 2, 1, 2),
        EnemyDefinition(ActorKind.ARMADILLO, 2, 1, 1),
    ),
    3: (
        EnemyDefinition(ActorKind.WARLOCK, 2, 1, 1),
        EnemyDefinition(ActorKind.BAT, 1, 1, 1),
    ),
    4: (
        EnemyDefinition(ActorKind.BLADEMASTER, 3, 1, 1),
        EnemyDefinition(ActorKind.SKELETON, 2, 1, 2),
    ),
}


def _carve_room(terrain: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    terrain[y : y + height, x : x + width] = Terrain.FLOOR


def _carve_corridor(
    terrain: np.ndarray, first: tuple[int, int], second: tuple[int, int]
) -> None:
    x1, y1 = first
    x2, y2 = second
    x_low, x_high = sorted((x1, x2))
    y_low, y_high = sorted((y1, y2))
    terrain[y1, x_low : x_high + 1] = Terrain.FLOOR
    terrain[y_low : y_high + 1, x2] = Terrain.FLOOR


def _room_level(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    terrain = np.full((height, width), Terrain.WALL, dtype=np.int16)
    room_count = int(rng.integers(4, 7))
    centres: list[tuple[int, int]] = []
    for _ in range(room_count):
        room_width = int(rng.integers(4, 8))
        room_height = int(rng.integers(4, 8))
        x = int(rng.integers(1, width - room_width - 1))
        y = int(rng.integers(1, height - room_height - 1))
        _carve_room(terrain, x, y, room_width, room_height)
        centre = (x + room_width // 2, y + room_height // 2)
        if centres:
            _carve_corridor(terrain, centres[-1], centre)
        centres.append(centre)
    return terrain


def _open_room(width: int, height: int) -> np.ndarray:
    terrain = np.full((height, width), Terrain.WALL, dtype=np.int16)
    terrain[3 : height - 3, 3 : width - 3] = Terrain.FLOOR
    return terrain


def _floor_positions(terrain: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.where(terrain == Terrain.FLOOR)
    return [(int(x), int(y)) for y, x in zip(ys, xs, strict=True)]


def generate_world(
    channels: RandomChannels,
    task: TaskSpec,
    zone: int,
    floor: int,
    *,
    player_health: int = 6,
    player_max_health: int = 6,
    gold: int = 0,
    bombs: int = 1,
    inventory: np.ndarray | None = None,
) -> WorldState:
    """Generate one deterministic level from a seed channel and level identity."""
    width = height = 25
    rng = channels.channel(f"level:{task.name}:{zone}:{floor}")
    if task.name in {"navigation", "single_enemy", "mixed_room"}:
        terrain = _open_room(width, height)
    else:
        terrain = _room_level(rng, width, height)

    positions = _floor_positions(terrain)
    player_position = min(positions, key=lambda p: p[0] + p[1])
    stairs = max(
        positions,
        key=lambda p: abs(p[0] - player_position[0]) + abs(p[1] - player_position[1]),
    )
    terrain[stairs[1], stairs[0]] = Terrain.STAIRS
    available = [p for p in positions if p not in {player_position, stairs}]
    rng.shuffle(available)

    is_boss_floor = task.include_boss and floor == task.regular_floors + 1
    if is_boss_floor:
        enemy_count = 1
    else:
        low, high = task.enemy_count
        enemy_count = int(rng.integers(low, high + 1)) if high else 0

    enemies: dict[int, Actor] = {}
    registry = ENEMY_REGISTRY[min(max(zone, 1), 4)]
    for index in range(min(enemy_count, len(available))):
        x, y = available.pop()
        if is_boss_floor:
            definition = EnemyDefinition(ActorKind.BOSS, 6 + zone * 2, 1, 1)
        else:
            definition = registry[index % len(registry)]
        entity_id = index + 2
        enemies[entity_id] = Actor(
            entity_id=entity_id,
            kind=definition.kind,
            x=x,
            y=y,
            health=definition.health,
            max_health=definition.health,
            damage=definition.damage,
            move_period=definition.move_period,
            facing=int((1, 3, 5, 7)[int(rng.integers(0, 4))]),
            boss=is_boss_floor,
        )

    traps = np.full((height, width), TrapKind.NONE, dtype=np.int16)
    if task.name in {"mixed_room", "floor", "zone", "all_zones"}:
        for x, y in available[-min(3, len(available)) :]:
            traps[y, x] = TrapKind.SPIKE

    items: dict[tuple[int, int], GroundItem] = {}
    if available and task.name not in {"navigation", "single_enemy"}:
        x, y = available[len(available) // 2]
        items[(x, y)] = GroundItem(ItemKind.GOLD, x, y, value=5)

    player = Actor(
        entity_id=1,
        kind=ActorKind.PLAYER,
        x=player_position[0],
        y=player_position[1],
        health=player_health,
        max_health=player_max_health,
    )
    if inventory is None:
        inventory = np.zeros((8, 3), dtype=np.int16)
        inventory[0] = (ItemKind.DAGGER, 1, 1)
        inventory[3] = (ItemKind.SHOVEL, 1, 1)
        if bombs:
            inventory[6] = (ItemKind.BOMB, bombs, 0)
    return WorldState(
        width=width,
        height=height,
        terrain=terrain,
        traps=traps,
        items=items,
        enemies=enemies,
        player=player,
        stairs=stairs,
        visible=np.zeros((height, width), dtype=bool),
        explored=np.zeros((height, width), dtype=bool),
        zone=zone,
        floor=floor,
        gold=gold,
        bombs=bombs,
        inventory=inventory.copy(),
    )

