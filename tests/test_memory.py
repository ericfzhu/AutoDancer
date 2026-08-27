from __future__ import annotations

import numpy as np
import pytest

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    MAP_SIZE,
    PLAYER_FEATURES,
    REVEALED_MAP_SIZE,
    GridChannel,
    MapChannel,
    PlayerFeature,
    Terrain,
)
from autodancer.memory import FloorMapMemory, MapCapacityError


def observation(x: int = 10, y: int = 20, *, zone: int = 1, floor: int = 1):
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16)
    player = np.zeros(PLAYER_FEATURES, dtype=np.int32)
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
    assert current[map_centre, map_centre, MapChannel.PLAYER] == 1
    assert current[map_centre, map_centre - 1, MapChannel.VISIT_COUNT] == 1


def test_floor_memory_accepts_full_game_reveal_without_marking_it_visited() -> None:
    memory = FloorMapMemory()
    value = observation()
    revealed = np.zeros((REVEALED_MAP_SIZE, REVEALED_MAP_SIZE), dtype=np.int8)
    revealed[4, 5] = Terrain.WALL
    result = memory.update(value, revealed)
    offset = (MAP_SIZE - REVEALED_MAP_SIZE) // 2
    assert result[offset + 4, offset + 5, MapChannel.TERRAIN_CLASS] == Terrain.WALL
    assert result[offset + 4, offset + 5, MapChannel.REVEAL_STATE] == 1
    assert result[offset + 4, offset + 5, MapChannel.VISIT_COUNT] == 0


def test_only_current_visibility_marks_a_tile_currently_visible() -> None:
    memory = FloorMapMemory()
    value = observation()
    centre = GRID_SIZE // 2
    value["grid"][centre, centre + 1, GridChannel.TERRAIN_CLASS] = Terrain.WALL
    value["grid"][centre, centre + 1, GridChannel.VISIBILITY] = 1
    result = memory.update(value)
    map_centre = MAP_SIZE // 2
    assert result[map_centre, map_centre + 1, MapChannel.REVEAL_STATE] == 1

    value["grid"][centre, centre + 1, GridChannel.VISIBILITY] = 2
    result = memory.update(value)
    assert result[map_centre, map_centre + 1, MapChannel.REVEAL_STATE] == 2


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


def test_floor_memory_rejects_level_bounds_that_would_clip() -> None:
    memory = FloorMapMemory()
    memory.update(observation(), map_bounds={"x": -22, "y": -12, "width": 65, "height": 65})
    memory.reset()
    with pytest.raises(MapCapacityError, match="66x65"):
        memory.update(observation(), map_bounds={"x": -23, "y": -12, "width": 66, "height": 65})


def test_floor_memory_retains_history_beyond_the_original_spawn_viewport() -> None:
    memory = FloorMapMemory()
    bounds = {"x": 10, "y": 20, "width": 65, "height": 65}
    memory.update(observation(), map_bounds=bounds)

    distant = observation(x=74)
    centre = GRID_SIZE // 2
    distant["grid"][centre, centre, GridChannel.TERRAIN_CLASS] = Terrain.FLOOR
    distant["grid"][centre, centre, GridChannel.VISIBILITY] = 2
    distant_view = memory.update(distant, map_bounds=bounds)
    map_centre = MAP_SIZE // 2
    assert distant_view[map_centre, map_centre, MapChannel.PLAYER] == 1
    assert distant_view[map_centre, map_centre, MapChannel.TERRAIN_CLASS] == Terrain.FLOOR

    memory.update(observation(), map_bounds=bounds)
    returned = memory.update(distant, map_bounds=bounds)
    assert returned[map_centre, map_centre, MapChannel.TERRAIN_CLASS] == Terrain.FLOOR
    assert returned[map_centre, map_centre, MapChannel.VISIT_COUNT] == 2


def test_bounds_anchored_reveal_covers_the_opposite_floor_extreme() -> None:
    memory = FloorMapMemory()
    bounds = {"x": 10, "y": 20, "width": 65, "height": 65}
    revealed = np.zeros((REVEALED_MAP_SIZE, REVEALED_MAP_SIZE), dtype=np.int8)
    revealed[64, 64] = Terrain.STAIRS
    result = memory.update(
        observation(x=10, y=20),
        revealed,
        bounds,
        {"x": 10, "y": 20},
    )
    centre = MAP_SIZE // 2
    assert result[centre + 64, centre + 64, MapChannel.TERRAIN_CLASS] == Terrain.STAIRS
    assert result[centre + 64, centre + 64, MapChannel.REVEAL_STATE] == 1
