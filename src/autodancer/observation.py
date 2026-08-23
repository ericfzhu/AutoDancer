"""Symbolic observation space exported by the Lua mod."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    MAP_CHANNELS,
    MAP_SIZE,
    PLAYER_FEATURES,
)


def observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(
                low=0,
                high=32767,
                shape=(GRID_SIZE, GRID_SIZE, GRID_CHANNELS),
                dtype=np.int16,
            ),
            "map_memory": spaces.Box(
                low=0,
                high=15,
                shape=(MAP_SIZE, MAP_SIZE, MAP_CHANNELS),
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
