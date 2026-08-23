"""Human-equivalent persistent map memory derived from revealed telemetry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from autodancer.constants import (
    GRID_SIZE,
    MAP_CHANNELS,
    MAP_SIZE,
    GridChannel,
    MapChannel,
    PlayerFeature,
)


class MapCapacityError(RuntimeError):
    """The fixed policy map cannot represent the current level without clipping."""


class FloorMapMemory:
    """Accumulate one floor's revealed terrain and Bard's traversal history.

    Absolute-coordinate history is retained for the whole floor. The policy sees
    a player-centred viewport containing only terrain marked revealed by the game,
    plus information Bard necessarily knows from its own actions. Dynamic
    off-screen entities are deliberately excluded.
    """

    def __init__(self) -> None:
        self._zone_floor: tuple[int, int] | None = None
        self._origin: tuple[int, int] | None = None
        self._terrain: dict[tuple[int, int], int] = {}
        self._last_seen: dict[tuple[int, int], int] = {}
        self._visits: dict[tuple[int, int], int] = {}
        self._last_visit: dict[tuple[int, int], int] = {}
        self._turn = 0

    def reset(self) -> None:
        self._zone_floor = None
        self._origin = None
        self._terrain.clear()
        self._last_seen.clear()
        self._visits.clear()
        self._last_visit.clear()
        self._turn = 0

    @staticmethod
    def _recency(age: int) -> int:
        if age <= 8:
            return 3
        if age <= 32:
            return 2
        return 1

    def _start_floor(self, zone: int, floor: int, x: int, y: int) -> None:
        self._zone_floor = (zone, floor)
        self._origin = (x, y)
        self._terrain.clear()
        self._last_seen.clear()
        self._visits.clear()
        self._last_visit.clear()
        self._turn = 0

    def _validate_capacity(self, map_bounds: Mapping[str, int] | None) -> None:
        if map_bounds is None:
            return
        width = int(map_bounds["width"])
        height = int(map_bounds["height"])
        if width > MAP_SIZE or height > MAP_SIZE:
            raise MapCapacityError(
                f"Level bounds are {width}x{height}; the policy's player-centred "
                f"map viewport supports levels up to {MAP_SIZE}x{MAP_SIZE}"
            )

    def update(
        self,
        observation: Mapping[str, np.ndarray],
        revealed_map: Sequence[Sequence[int]] | None = None,
        map_bounds: Mapping[str, int] | None = None,
    ) -> np.ndarray:
        player = observation["player"]
        x, y = int(player[PlayerFeature.X]), int(player[PlayerFeature.Y])
        zone, floor = int(player[PlayerFeature.ZONE]), int(player[PlayerFeature.FLOOR])
        if self._zone_floor != (zone, floor) or self._origin is None:
            self._start_floor(zone, floor, x, y)
        self._validate_capacity(map_bounds)
        self._turn += 1

        grid = observation["grid"]
        centre = GRID_SIZE // 2
        for row, column in np.argwhere(grid[..., GridChannel.VISIBILITY] > 0):
            position = (x + int(column) - centre, y + int(row) - centre)
            self._terrain[position] = int(grid[row, column, GridChannel.TERRAIN_CLASS])
            self._last_seen[position] = self._turn

        if revealed_map is not None:
            origin_x, origin_y = self._origin
            half = MAP_SIZE // 2
            raw = np.asarray(revealed_map)
            if raw.shape != (MAP_SIZE, MAP_SIZE):
                raise ValueError(
                    f"revealed_map has shape {raw.shape}; expected {(MAP_SIZE, MAP_SIZE)}"
                )
            for row, column in np.argwhere(raw > 0):
                position = (origin_x + int(column) - half, origin_y + int(row) - half)
                self._terrain[position] = int(raw[row, column])
                self._last_seen[position] = self._turn

        position = (x, y)
        self._visits[position] = self._visits.get(position, 0) + 1
        self._last_visit[position] = self._turn
        return self.observation(position)

    def observation(self, player_position: tuple[int, int]) -> np.ndarray:
        result = np.zeros((MAP_SIZE, MAP_SIZE, MAP_CHANNELS), dtype=np.int16)
        if self._origin is None:
            return result
        origin_x, origin_y = player_position
        half = MAP_SIZE // 2
        for x, y in self._terrain.keys() | self._visits.keys():
            column, row = x - origin_x + half, y - origin_y + half
            if not (0 <= row < MAP_SIZE and 0 <= column < MAP_SIZE):
                continue
            if (x, y) in self._terrain:
                result[row, column, MapChannel.TERRAIN_CLASS] = self._terrain[(x, y)]
                result[row, column, MapChannel.REVEAL_STATE] = (
                    2 if self._last_seen[(x, y)] == self._turn else 1
                )
            visits = self._visits.get((x, y), 0)
            result[row, column, MapChannel.VISIT_COUNT] = min(visits, 15)
            if visits:
                result[row, column, MapChannel.VISIT_RECENCY] = self._recency(
                    self._turn - self._last_visit[(x, y)]
                )
        result[half, half, MapChannel.PLAYER] = 1
        return result
