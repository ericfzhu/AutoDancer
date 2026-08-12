"""Shared symbolic observation schema."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    PLAYER_FEATURES,
    Action,
    ActorKind,
    GridChannel,
    StatusFlag,
    Terrain,
)
from autodancer.model import WorldState


def observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(
                low=0,
                high=32767,
                shape=(GRID_SIZE, GRID_SIZE, GRID_CHANNELS),
                dtype=np.int16,
            ),
            "player": spaces.Box(
                low=-2**31,
                high=2**31 - 1,
                shape=(PLAYER_FEATURES,),
                dtype=np.int32,
            ),
            "inventory": spaces.Box(
                low=0,
                high=32767,
                shape=(INVENTORY_SLOTS, INVENTORY_FEATURES),
                dtype=np.int16,
            ),
            "action_mask": spaces.Box(0, 1, shape=(ACTION_COUNT,), dtype=np.int8),
        }
    )


def _line_visible(state: WorldState, x1: int, y1: int, x2: int, y2: int) -> bool:
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    error = dx - dy
    x, y = x1, y1
    while (x, y) != (x2, y2):
        doubled = error * 2
        if doubled > -dy:
            error -= dy
            x += sx
        if doubled < dx:
            error += dx
            y += sy
        if (x, y) != (x2, y2) and state.terrain[y, x] == Terrain.WALL:
            return False
    return True


def update_visibility(state: WorldState, radius: int = GRID_SIZE // 2) -> int:
    previous = state.explored.copy()
    state.visible.fill(False)
    px, py = state.player.position
    for y in range(max(0, py - radius), min(state.height, py + radius + 1)):
        for x in range(max(0, px - radius), min(state.width, px + radius + 1)):
            if max(abs(x - px), abs(y - py)) <= radius and _line_visible(
                state, px, py, x, y
            ):
                state.visible[y, x] = True
    state.explored |= state.visible
    return int(np.count_nonzero(state.explored & ~previous))


def action_mask(state: WorldState) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    for action in (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT):
        mask[action] = 1
    mask[Action.WAIT] = 1
    mask[Action.BOMB] = int(state.bombs > 0)
    mask[Action.ITEM_1] = int(state.inventory[1, 0] != 0)
    mask[Action.ITEM_2] = int(state.inventory[2, 0] != 0)
    mask[Action.THROW] = int(state.weapon_damage > 0)
    mask[Action.SPELL_1] = int(state.inventory[4, 0] != 0)
    mask[Action.SPELL_2] = int(state.inventory[5, 0] != 0)
    return mask


def encode_observation(state: WorldState, task_index: int = 0) -> dict[str, np.ndarray]:
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16)
    radius = GRID_SIZE // 2
    px, py = state.player.position
    enemies_by_position = {enemy.position: enemy for enemy in state.enemies.values()}
    bombs_by_position = {(bomb.x, bomb.y) for bomb in state.active_bombs}

    for gy in range(GRID_SIZE):
        world_y = py + gy - radius
        for gx in range(GRID_SIZE):
            world_x = px + gx - radius
            if not state.in_bounds(world_x, world_y) or not state.explored[world_y, world_x]:
                continue
            visible = state.visible[world_y, world_x]
            grid[gy, gx, GridChannel.TERRAIN] = state.terrain[world_y, world_x]
            grid[gy, gx, GridChannel.VISIBILITY] = 2 if visible else 1
            if not visible:
                continue
            if (world_x, world_y) == state.player.position:
                grid[gy, gx, GridChannel.ACTOR] = ActorKind.PLAYER
                grid[gy, gx, GridChannel.HEALTH] = state.player.health
            elif enemy := enemies_by_position.get((world_x, world_y)):
                grid[gy, gx, GridChannel.ACTOR] = enemy.kind
                grid[gy, gx, GridChannel.HEALTH] = enemy.health
            item = state.items.get((world_x, world_y))
            if item is not None:
                grid[gy, gx, GridChannel.ITEM] = item.kind
            grid[gy, gx, GridChannel.TRAP] = state.traps[world_y, world_x]
            status = StatusFlag.NONE
            if (world_x, world_y) in bombs_by_position:
                status = StatusFlag.BOMB
            elif (world_x, world_y) == state.stairs and state.enemies:
                status = StatusFlag.EXIT_LOCKED
            grid[gy, gx, GridChannel.STATUS] = status

    player = np.zeros(PLAYER_FEATURES, dtype=np.int32)
    player[:] = (
        state.player.health,
        state.player.max_health,
        state.gold,
        state.groove,
        state.player.x,
        state.player.y,
        state.zone,
        state.floor,
        state.turn,
        state.bombs,
        state.weapon_damage,
        len(state.enemies),
        int(state.player.position == state.stairs),
        task_index,
        int(state.won),
        int(state.dead),
    )
    return {
        "grid": grid,
        "player": player,
        "inventory": state.inventory.astype(np.int16, copy=True),
        "action_mask": action_mask(state),
    }
