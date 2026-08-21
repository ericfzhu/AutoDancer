from __future__ import annotations

import numpy as np

from autodancer.constants import (
    GRID_CHANNELS,
    INVENTORY_FEATURES,
    Action,
    ActorKind,
    GridChannel,
    Terrain,
)
from autodancer.live.explore import LiveExplorer


def observation() -> dict[str, np.ndarray]:
    grid = np.zeros((21, 21, GRID_CHANNELS), dtype=np.int16)
    grid[10, 10, GridChannel.TERRAIN_CLASS] = Terrain.FLOOR
    grid[10, 10, GridChannel.VISIBILITY] = 2
    player = np.zeros(16, dtype=np.int32)
    action_mask = np.zeros(11, dtype=np.int8)
    action_mask[:4] = 1
    return {
        "grid": grid,
        "player": player,
        "inventory": np.zeros((8, INVENTORY_FEATURES), dtype=np.int16),
        "action_mask": action_mask,
    }


def test_explorer_attacks_adjacent_visible_enemy() -> None:
    value = observation()
    value["grid"][10, 11, GridChannel.TERRAIN_CLASS] = Terrain.FLOOR
    value["grid"][10, 11, GridChannel.VISIBILITY] = 2
    value["grid"][10, 11, GridChannel.ACTOR_CLASS] = ActorKind.GREEN_SLIME
    assert LiveExplorer().choose(value) == Action.RIGHT


def test_explorer_advances_into_unexplored_frontier() -> None:
    assert LiveExplorer().choose(observation()) == Action.UP


def test_explorer_prioritizes_visible_stairs() -> None:
    value = observation()
    value["grid"][10, 11, GridChannel.TERRAIN_CLASS] = Terrain.STAIRS
    value["grid"][10, 11, GridChannel.VISIBILITY] = 2
    explorer = LiveExplorer()
    assert explorer.choose(value) == Action.RIGHT
    assert explorer.last_reason == "stairs"


def test_route_preserves_zero_valued_up_as_first_action() -> None:
    explorer = LiveExplorer()
    explorer.terrain = {
        (0, 0): Terrain.FLOOR,
        (0, -1): Terrain.FLOOR,
        (1, -1): Terrain.STAIRS,
    }
    assert explorer._route((0, 0), {(1, -1)}, allow_final_unknown=False) == Action.UP


def test_reset_level_discards_previous_floor_routing_state() -> None:
    explorer = LiveExplorer()
    explorer.terrain[(4, 5)] = Terrain.STAIRS
    explorer.traps[(4, 5)] = 1
    explorer.attempted_dig.add((3, 5))
    explorer.reset_level()
    assert explorer.terrain == {}
    assert explorer.traps == {}
    assert explorer.attempted_dig == set()


def test_explorer_rotates_fallback_actions_after_frontiers_are_exhausted() -> None:
    value = observation()
    explorer = LiveExplorer()
    explorer.update(value)
    player = (0, 0)
    for x in range(-2, 3):
        for y in range(-2, 3):
            if (x, y) != player:
                explorer.terrain[x, y] = Terrain.WALL
    assert explorer.choose(value) == Action.UP
    assert explorer.choose(value) == Action.RIGHT


def test_explorer_attempts_an_indestructible_wall_only_once() -> None:
    value = observation()
    value["grid"][9, 10, GridChannel.TERRAIN_CLASS] = Terrain.WALL
    value["grid"][9, 10, GridChannel.VISIBILITY] = 2
    explorer = LiveExplorer()
    assert explorer.choose(value) == Action.RIGHT
    explorer.attempted_unknown.update(
        {
            ((0, 0), (1, 0)),
            ((0, 0), (0, 1)),
            ((0, 0), (-1, 0)),
        }
    )
    assert explorer.choose(value) == Action.UP
    assert explorer.choose(value) != Action.UP


def test_explorer_does_not_route_through_a_known_bounce_trap() -> None:
    value = observation()
    value["grid"][10, 11, GridChannel.TERRAIN_CLASS] = Terrain.FLOOR
    value["grid"][10, 11, GridChannel.TRAP] = 2
    value["grid"][10, 11, GridChannel.VISIBILITY] = 2
    value["grid"][10, 12, GridChannel.TERRAIN_CLASS] = Terrain.FLOOR
    value["grid"][10, 12, GridChannel.ACTOR_CLASS] = ActorKind.GREEN_SLIME
    value["grid"][10, 12, GridChannel.VISIBILITY] = 2
    assert LiveExplorer().choose(value) != Action.RIGHT


def test_explorer_does_not_revisit_an_exhausted_frontier() -> None:
    value = observation()
    value["grid"][10, 11, GridChannel.TERRAIN_CLASS] = Terrain.FLOOR
    value["grid"][10, 11, GridChannel.VISIBILITY] = 2
    explorer = LiveExplorer()
    explorer.update(value)
    for position in ((0, 0), (1, 0)):
        for neighbor in explorer._neighbors(position):
            if neighbor not in explorer.terrain:
                explorer.attempted_unknown.add((position, neighbor))
    assert explorer.choose(value) != Action.RIGHT
