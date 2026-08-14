"""Bounded heuristic controller for autonomous live conformance collection."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from autodancer.constants import DIRECTION_DELTAS, Action, GridChannel, Terrain
from autodancer.envs.live import AutoDancerLiveEnv

_ACTIONS = (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)


class LiveExplorer:
    """Explore revealed terrain and engage visible enemies without hidden state."""

    def __init__(self) -> None:
        self.reset_level()

    def reset_level(self) -> None:
        """Forget coordinates and routing state when the game loads a new floor."""
        self.terrain: dict[tuple[int, int], int] = {}
        self.traps: dict[tuple[int, int], int] = {}
        self.attempted_unknown: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.attempted_dig: set[tuple[int, int]] = set()
        self.fallback_index = 0
        self.frontier_target: tuple[int, int] | None = None
        self.retired_frontiers: set[tuple[int, int]] = set()
        self.previous_position: tuple[int, int] | None = None
        self.last_reason = ""

    def update(self, observation: dict[str, np.ndarray]) -> tuple[int, int]:
        grid = observation["grid"]
        player = observation["player"]
        px, py = int(player[4]), int(player[5])
        radius = grid.shape[0] // 2
        for row, column in np.argwhere(grid[..., GridChannel.VISIBILITY] > 0):
            position = px + int(column) - radius, py + int(row) - radius
            self.terrain[position] = int(grid[row, column, GridChannel.TERRAIN])
            self.traps[position] = int(grid[row, column, GridChannel.TRAP])
        return px, py

    def choose(self, observation: dict[str, np.ndarray]) -> Action:
        player = self.update(observation)
        previous_position = self.previous_position
        self.previous_position = player
        enemies = self._visible_enemies(observation)
        adjacent = self._adjacent_action(player, enemies)
        if adjacent is not None:
            self.last_reason = "adjacent_enemy"
            return adjacent

        action = self._route(player, enemies, allow_final_unknown=False)
        if action is not None:
            self.last_reason = "visible_enemy"
            return action

        stairs = {
            position
            for position, terrain in self.terrain.items()
            if terrain == Terrain.STAIRS
        }
        action = self._route(player, stairs, allow_final_unknown=False)
        if action is not None:
            self.last_reason = "stairs"
            return action

        frontiers = {
            position
            for position, terrain in self.terrain.items()
            if terrain in {Terrain.FLOOR, Terrain.STAIRS}
            and position not in self.retired_frontiers
            and self.traps.get(position, 0) == 0
            and any(
                neighbor not in self.terrain
                and (position, neighbor) not in self.attempted_unknown
                for neighbor in self._neighbors(position)
            )
        }
        if self.frontier_target not in frontiers:
            if self.frontier_target is not None:
                self.retired_frontiers.add(self.frontier_target)
            self.frontier_target = None
        if player in frontiers:
            for action, neighbor in self._action_neighbors(player):
                edge = player, neighbor
                if neighbor not in self.terrain and edge not in self.attempted_unknown:
                    self.attempted_unknown.add(edge)
                    self.frontier_target = None
                    self.last_reason = "unknown"
                    return action

        if self.frontier_target is None:
            ordered_frontiers = sorted(
                frontiers,
                key=lambda position: (
                    abs(position[0] - player[0]) + abs(position[1] - player[1]),
                    position,
                ),
            )
            for target in ordered_frontiers:
                action = self._route(player, {target}, allow_final_unknown=False)
                if action is not None:
                    if self._action_target(player, action) == previous_position:
                        self.retired_frontiers.add(target)
                        continue
                    self.frontier_target = target
                    self.last_reason = "frontier"
                    return action
        else:
            action = self._route(
                player, {self.frontier_target}, allow_final_unknown=False
            )
            if action is not None:
                if self._action_target(player, action) != previous_position:
                    self.last_reason = "frontier"
                    return action
                self.retired_frontiers.add(self.frontier_target)
            self.frontier_target = None

        dig_targets = {
            position
            for position, terrain in self.terrain.items()
            if terrain == Terrain.WALL
            and position not in self.attempted_dig
            and any(neighbor not in self.terrain for neighbor in self._neighbors(position))
        }
        action = self._route(player, dig_targets, allow_final_unknown=True)
        if action is not None:
            dx, dy = DIRECTION_DELTAS[action]
            adjacent = player[0] + dx, player[1] + dy
            if adjacent in dig_targets:
                self.attempted_dig.add(adjacent)
            self.last_reason = "dig"
            return action

        for offset in range(len(_ACTIONS)):
            index = (self.fallback_index + offset) % len(_ACTIONS)
            action = _ACTIONS[index]
            dx, dy = DIRECTION_DELTAS[action]
            target = player[0] + dx, player[1] + dy
            if (
                observation["action_mask"][int(action)]
                and target not in self.attempted_dig
                and self.traps.get(target, 0) == 0
            ):
                self.fallback_index = (index + 1) % len(_ACTIONS)
                self.last_reason = "fallback"
                return action
        raise RuntimeError("The live observation masks every movement action")

    def _visible_enemies(
        self, observation: dict[str, np.ndarray]
    ) -> set[tuple[int, int]]:
        grid = observation["grid"]
        px, py = (int(value) for value in observation["player"][4:6])
        radius = grid.shape[0] // 2
        result: set[tuple[int, int]] = set()
        for row, column in np.argwhere(grid[..., GridChannel.ACTOR] > 1):
            result.add((px + int(column) - radius, py + int(row) - radius))
        return result

    def _route(
        self,
        start: tuple[int, int],
        goals: set[tuple[int, int]],
        *,
        allow_final_unknown: bool,
    ) -> Action | None:
        if not goals or start in goals:
            return None
        queue = deque([(start, None)])
        visited = {start}
        while queue:
            position, first_action = queue.popleft()
            for action, neighbor in self._action_neighbors(position):
                if neighbor in visited:
                    continue
                is_goal = neighbor in goals
                terrain = self.terrain.get(neighbor, Terrain.UNKNOWN)
                walkable = (
                    terrain in {Terrain.FLOOR, Terrain.STAIRS}
                    and self.traps.get(neighbor, 0) == 0
                )
                if not walkable and not (is_goal and allow_final_unknown):
                    continue
                visited.add(neighbor)
                candidate = first_action if first_action is not None else action
                if is_goal:
                    return candidate
                queue.append((neighbor, candidate))
        return None

    @staticmethod
    def _neighbors(position: tuple[int, int]):
        x, y = position
        for dx, dy in DIRECTION_DELTAS.values():
            yield x + dx, y + dy

    @staticmethod
    def _action_neighbors(position: tuple[int, int]):
        x, y = position
        for action in _ACTIONS:
            dx, dy = DIRECTION_DELTAS[action]
            yield action, (x + dx, y + dy)

    @staticmethod
    def _action_target(position: tuple[int, int], action: Action) -> tuple[int, int]:
        dx, dy = DIRECTION_DELTAS[action]
        return position[0] + dx, position[1] + dy

    @staticmethod
    def _adjacent_action(
        player: tuple[int, int], enemies: set[tuple[int, int]]
    ) -> Action | None:
        return next(
            (
                action
                for action, position in LiveExplorer._action_neighbors(player)
                if position in enemies
            ),
            None,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomously collect a bounded live trace")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--max-turns", type=int, default=100)
    arguments = parser.parse_args()
    environment = AutoDancerLiveEnv(log_path=arguments.log_path, attach_existing=True)
    observation, info = environment.reset()
    explorer = LiveExplorer()
    floor = info.get("floor")
    for turn in range(1, arguments.max_turns + 1):
        action = explorer.choose(observation)
        observation, reward, terminated, truncated, info = environment.step(action)
        next_floor = info.get("floor")
        if next_floor != floor:
            explorer.reset_level()
            floor = next_floor
        print(
            f"{turn}: {action.name} ({explorer.last_reason}) "
            f"floor={floor} "
            f"pos={tuple(observation['player'][4:6])} "
            f"hp={int(observation['player'][0])} reward={reward:.3f} "
            f"events={info['raw_events']}"
        )
        if terminated or truncated:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
