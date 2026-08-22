"""Stateful, bounded reward shaping for live Bard runs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autodancer.constants import GridChannel, PlayerFeature, Terrain

REWARD_PROFILE_VERSION = 2


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for the default progress-first reward profile."""

    turn: float = -0.005
    new_position: float = 0.015
    revisit: float = -0.01
    new_tile: float = 0.001
    max_new_tiles_per_turn: int = 25
    enemy_damage: float = 0.025
    max_rewarded_damage_per_enemy: int = 16
    enemy_kill: float = 0.25
    player_damage: float = -0.15
    new_item_type: float = 0.15
    currency: float = 0.002
    max_currency_per_turn: int = 25
    container_opened: float = 0.05
    stairs_discovered: float = 0.5
    stair_progress: float = 0.05
    max_stair_distance_delta: int = 4
    floor_complete: float = 5.0
    zone_complete: float = 10.0
    victory: float = 50.0
    death: float = -2.0
    aborted: float = -1.0

    def __post_init__(self) -> None:
        for name in (
            "max_new_tiles_per_turn",
            "max_rewarded_damage_per_enemy",
            "max_currency_per_turn",
            "max_stair_distance_delta",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def specification(self) -> dict[str, Any]:
        return {"version": REWARD_PROFILE_VERSION, "weights": asdict(self)}


class RewardTracker:
    """Episode-local reward state that prevents repeatable shaping loops."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.seen_tiles: set[tuple[int, int, int, int]] = set()
        self.visited_positions: set[tuple[int, int, int, int]] = set()
        self.seen_item_types: set[int] = set()
        self.rewarded_kills: set[int] = set()
        self.rewarded_damage: dict[int, int] = {}
        self.known_stairs: set[tuple[int, int, int, int]] = set()
        self.stair_distance: int | None = None
        self.zone = 0
        self.floor = 0

    @staticmethod
    def _position(observation: Mapping[str, np.ndarray]) -> tuple[int, int]:
        player = observation["player"]
        return int(player[PlayerFeature.X]), int(player[PlayerFeature.Y])

    @classmethod
    def _revealed_tiles(
        cls,
        observation: Mapping[str, np.ndarray],
        zone: int,
        floor: int,
    ) -> set[tuple[int, int, int, int]]:
        grid = observation["grid"]
        px, py = cls._position(observation)
        centre = grid.shape[0] // 2
        return {
            (zone, floor, px + int(column) - centre, py + int(row) - centre)
            for row, column in np.argwhere(grid[..., GridChannel.VISIBILITY] > 0)
        }

    @staticmethod
    def _inventory_types(observation: Mapping[str, np.ndarray]) -> set[int]:
        return {int(value) for value in observation["inventory"][:, 1] if int(value) != 0}

    @classmethod
    def _stairs(
        cls,
        observation: Mapping[str, np.ndarray],
        zone: int,
        floor: int,
    ) -> set[tuple[int, int, int, int]]:
        grid = observation["grid"]
        px, py = cls._position(observation)
        centre = grid.shape[0] // 2
        return {
            (zone, floor, px + int(column) - centre, py + int(row) - centre)
            for row, column in np.argwhere(
                grid[..., GridChannel.TERRAIN_CLASS] == int(Terrain.STAIRS)
            )
        }

    @staticmethod
    def _nearest_stair_distance(
        position: tuple[int, int],
        stairs: set[tuple[int, int, int, int]],
        zone: int,
        floor: int,
    ) -> int | None:
        distances = [
            abs(position[0] - stair_x) + abs(position[1] - stair_y)
            for stair_zone, stair_floor, stair_x, stair_y in stairs
            if stair_zone == zone and stair_floor == floor
        ]
        return min(distances) if distances else None

    def reset(
        self, observation: Mapping[str, np.ndarray], info: Mapping[str, Any]
    ) -> None:
        self.zone = int(info.get("zone") or 0)
        self.floor = int(info.get("floor") or 0)
        self.seen_tiles = self._revealed_tiles(observation, self.zone, self.floor)
        x, y = self._position(observation)
        self.visited_positions = {(self.zone, self.floor, x, y)}
        self.seen_item_types = self._inventory_types(observation)
        self.known_stairs = self._stairs(observation, self.zone, self.floor)
        self.stair_distance = self._nearest_stair_distance(
            (x, y), self.known_stairs, self.zone, self.floor
        )
        self.rewarded_kills.clear()
        self.rewarded_damage.clear()

    def score(
        self,
        observation: Mapping[str, np.ndarray],
        info: Mapping[str, Any],
        events: Iterable[Mapping[str, Any]],
        *,
        terminated: bool,
        truncated: bool,
    ) -> tuple[float, dict[str, float]]:
        config = self.config
        components: dict[str, float] = {"turn": config.turn}

        zone = int(info.get("zone") or self.zone)
        floor = int(info.get("floor") or self.floor)
        x, y = self._position(observation)
        position = (zone, floor, x, y)
        same_floor = zone == self.zone and floor == self.floor
        if position not in self.visited_positions:
            components["new_position"] = config.new_position
            self.visited_positions.add(position)
        else:
            components["revisit"] = config.revisit

        revealed = self._revealed_tiles(observation, zone, floor)
        new_tiles = len(revealed - self.seen_tiles)
        if new_tiles:
            credited = min(new_tiles, config.max_new_tiles_per_turn)
            components["new_tile"] = config.new_tile * credited
            self.seen_tiles.update(revealed)

        item_types = self._inventory_types(observation)
        new_items = item_types - self.seen_item_types
        if new_items:
            components["new_item_type"] = config.new_item_type * len(new_items)
            self.seen_item_types.update(item_types)

        observed_stairs = self._stairs(observation, zone, floor)
        if not same_floor:
            self.known_stairs = observed_stairs
            self.stair_distance = self._nearest_stair_distance(
                (x, y), self.known_stairs, zone, floor
            )
        else:
            new_stairs = observed_stairs - self.known_stairs
            if new_stairs:
                components["stairs_discovered"] = config.stairs_discovered * len(new_stairs)
                self.known_stairs.update(new_stairs)
            distance = self._nearest_stair_distance((x, y), self.known_stairs, zone, floor)
            # Discovery changes the potential function itself, so establish a new
            # baseline instead of paying both discovery and artificial progress.
            if not new_stairs and distance is not None and self.stair_distance is not None:
                delta = self.stair_distance - distance
                credited = max(
                    -config.max_stair_distance_delta,
                    min(delta, config.max_stair_distance_delta),
                )
                if credited:
                    components["stair_progress"] = config.stair_progress * credited
            self.stair_distance = distance

        if zone > self.zone:
            components["zone_complete"] = config.zone_complete * (zone - self.zone)
        elif zone == self.zone and floor > self.floor:
            components["floor_complete"] = config.floor_complete * (floor - self.floor)
        self.zone, self.floor = zone, floor

        for event in events:
            self._score_event(event, components)

        status = str(info.get("episode_status", "running"))
        if terminated and status == "won":
            components["victory"] = config.victory
        elif terminated and status == "dead":
            components["death"] = config.death
        elif truncated:
            components["aborted"] = config.aborted

        return float(sum(components.values())), components

    def _score_event(
        self, event: Mapping[str, Any], components: dict[str, float]
    ) -> None:
        config = self.config
        kind = str(event.get("kind", ""))
        amount = max(int(event.get("amount", 0) or 0), 0)
        entity_id = max(int(event.get("entity_id", 0) or 0), 0)
        if kind == "enemy_damage":
            previously = self.rewarded_damage.get(entity_id, 0) if entity_id else 0
            credited = min(amount, max(config.max_rewarded_damage_per_enemy - previously, 0))
            if credited:
                components["enemy_damage"] = components.get("enemy_damage", 0.0) + (
                    config.enemy_damage * credited
                )
                if entity_id:
                    self.rewarded_damage[entity_id] = previously + credited
        elif kind == "enemy_kill" and (entity_id == 0 or entity_id not in self.rewarded_kills):
            components["enemy_kill"] = components.get("enemy_kill", 0.0) + config.enemy_kill
            if entity_id:
                self.rewarded_kills.add(entity_id)
        elif kind == "player_damage":
            components["player_damage"] = components.get("player_damage", 0.0) + (
                config.player_damage * amount
            )
        elif kind == "item_collected":
            components["currency"] = components.get("currency", 0.0) + (
                config.currency * min(amount, config.max_currency_per_turn)
            )
        elif kind == "container_opened":
            components["container_opened"] = components.get("container_opened", 0.0) + (
                config.container_opened
            )


def load_reward_config(path: str | Path | None) -> RewardConfig:
    if path is None:
        return RewardConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reward configuration must be a JSON object")
    unknown = set(payload) - set(asdict(RewardConfig()))
    if unknown:
        raise ValueError(f"Unknown reward configuration fields: {sorted(unknown)}")
    return RewardConfig(**payload)


def reward_from_event_dicts(
    events: Iterable[Mapping[str, Any]], values: Mapping[str, float] | None = None
) -> float:
    """Compatibility helper for isolated event tests; live environments use RewardTracker."""
    config = RewardConfig(**dict(values or {}))
    components: dict[str, float] = {"turn": config.turn}
    tracker = RewardTracker(config)
    for event in events:
        tracker._score_event(event, components)
    return float(sum(components.values()))
