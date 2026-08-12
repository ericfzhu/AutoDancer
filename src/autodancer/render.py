"""Small generated renderer. It does not use game assets."""

from __future__ import annotations

import numpy as np

from autodancer.constants import GRID_SIZE, RGB_SIZE, ActorKind, GridChannel, Terrain

TERRAIN_COLOURS = {
    Terrain.UNKNOWN: (4, 5, 8),
    Terrain.FLOOR: (38, 36, 48),
    Terrain.WALL: (82, 72, 98),
    Terrain.STAIRS: (170, 130, 40),
}
ACTOR_COLOURS = {
    ActorKind.PLAYER: (70, 220, 245),
    ActorKind.GREEN_SLIME: (65, 205, 90),
    ActorKind.BLUE_SLIME: (70, 100, 230),
    ActorKind.SKELETON: (225, 225, 210),
    ActorKind.BAT: (175, 70, 200),
    ActorKind.ARMADILLO: (195, 125, 55),
    ActorKind.WARLOCK: (225, 55, 180),
    ActorKind.BLADEMASTER: (225, 80, 70),
    ActorKind.BOSS: (250, 45, 45),
}


def render_grid(grid: np.ndarray) -> np.ndarray:
    tile_size = RGB_SIZE // GRID_SIZE
    offset = (RGB_SIZE - tile_size * GRID_SIZE) // 2
    image = np.full((RGB_SIZE, RGB_SIZE, 3), (8, 8, 12), dtype=np.uint8)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            terrain = Terrain(int(grid[y, x, GridChannel.TERRAIN]))
            colour = TERRAIN_COLOURS[terrain]
            y0, x0 = offset + y * tile_size, offset + x * tile_size
            image[y0 : y0 + tile_size, x0 : x0 + tile_size] = colour
            if grid[y, x, GridChannel.TRAP]:
                image[y0 + 3 : y0 + 7, x0 + 3 : x0 + 7] = (220, 80, 65)
            if grid[y, x, GridChannel.ITEM]:
                image[y0 + 4 : y0 + 8, x0 + 4 : x0 + 8] = (255, 215, 60)
            actor_value = int(grid[y, x, GridChannel.ACTOR])
            if actor_value:
                actor = ActorKind(actor_value)
                image[y0 + 2 : y0 + 10, x0 + 2 : x0 + 10] = ACTOR_COLOURS[actor]
    return image

