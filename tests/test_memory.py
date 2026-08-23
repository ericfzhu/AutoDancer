from __future__ import annotations

import numpy as np

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    MAP_SIZE,
    GridChannel,
    MapChannel,
    PlayerFeature,
    Terrain,
)
from autodancer.memory import FloorMapMemory


def observation(x: int = 10, y: int = 20, *, zone: int = 1, floor: int = 1):
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16)
    player = np.zeros(16, dtype=np.int32)
    player[PlayerFeature.X] = x
    player[PlayerFeature.Y] = y
    player[PlayerFeature.ZONE] = zone
    player[PlayerFeature.FLOOR] = floor
    return {
        "grid": grid,
        "player": player,
        "inventory": np.zeros((INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16),
        "action_mask": np.ones(ACTION_COUNT, dtype=np.int8),
    }


def test_floor_memory_accumulates_revealed_terrain_and_visits() -> None:
    memory = FloorMapMemory()
    first = observation()
    centre = GRID_SIZE // 2
    first["grid"][centre, centre, GridChannel.TERRAIN_CLASS] = Terrain.FLOOR
    first["grid"][centre, centre, GridChannel.VISIBILITY] = 2
    initial = memory.update(first)
    map_centre = MAP_SIZE // 2
    assert initial[map_centre, map_centre, MapChannel.TERRAIN_CLASS] == Terrain.FLOOR
    assert initial[map_centre, map_centre, MapChannel.VISIT_COUNT] == 1
    assert initial[map_centre, map_centre, MapChannel.PLAYER] == 1

    moved = observation(x=11)
    moved["grid"][centre, centre, GridChannel.TERRAIN_CLASS] = Terrain.FLOOR
    moved["grid"][centre, centre, GridChannel.VISIBILITY] = 2
    current = memory.update(moved)
    assert current[map_centre, map_centre, MapChannel.VISIT_COUNT] == 1
    assert current[map_centre, map_centre, MapChannel.PLAYER] == 0
    assert current[map_centre, map_centre + 1, MapChannel.PLAYER] == 1


def test_floor_memory_accepts_full_game_reveal_without_marking_it_visited() -> None:
    memory = FloorMapMemory()
    value = observation()
    revealed = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    revealed[4, 5] = Terrain.WALL
    result = memory.update(value, revealed)
    assert result[4, 5, MapChannel.TERRAIN_CLASS] == Terrain.WALL
    assert result[4, 5, MapChannel.REVEAL_STATE] == 2
    assert result[4, 5, MapChannel.VISIT_COUNT] == 0


def test_floor_transition_clears_previous_map() -> None:
    memory = FloorMapMemory()
    first = observation()
    centre = GRID_SIZE // 2
    first["grid"][centre, centre + 1, GridChannel.TERRAIN_CLASS] = Terrain.WALL
    first["grid"][centre, centre + 1, GridChannel.VISIBILITY] = 1
    memory.update(first)
    next_floor = memory.update(observation(floor=2))
    map_centre = MAP_SIZE // 2
    assert next_floor[map_centre, map_centre + 1, MapChannel.TERRAIN_CLASS] == Terrain.UNKNOWN
    assert next_floor[map_centre, map_centre, MapChannel.VISIT_COUNT] == 1
