"""Policy-side action contracts, including episode-local invalid-action memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autodancer.constants import (
    DIRECTION_DELTAS,
    GRID_SIZE,
    Action,
    ActorKind,
    GridChannel,
    MapChannel,
    PlayerFeature,
    Terrain,
)

ACTION_CONTRACTS = (
    "bounded-bard-v1",
    "bounded-bard-navigation-v1",
    "current",
    "legacy-no-wait",
    "known-invalid-wall-v1",
    "map-navigation-prior-v1",
    "map-navigation-prior-v2",
    "dagger-controls-v1",
)
_WALL_MEMORY_CONTRACTS = {
    "known-invalid-wall-v1",
    "map-navigation-prior-v1",
    "map-navigation-prior-v2",
    "dagger-controls-v1",
}

_PLAYER_CONTROL_CHANNELS = (
    GridChannel.STATUS,
    GridChannel.FACING,
    GridChannel.BEAT_DELAY,
    GridChannel.BEAT_INTERVAL,
    GridChannel.FROZEN_TURNS,
    GridChannel.CONFUSED_TURNS,
    GridChannel.CHARGE_STATE,
    GridChannel.CHARGE_DIRECTION,
    GridChannel.SHIELD_DIRECTION,
)


class UnsupportedLoadoutError(RuntimeError):
    """The episode escaped the declared research scope; never a gameplay loss."""


def _state_signature(observation: dict[str, np.ndarray], action: Action) -> tuple[int, ...]:
    """Describe everything observed that could make one wall action change meaning."""
    if action not in DIRECTION_DELTAS:
        raise ValueError("Only directional actions have wall-state signatures")
    player = observation["player"]
    dx, dy = DIRECTION_DELTAS[action]
    centre = GRID_SIZE // 2
    target = observation["grid"][centre + dy, centre + dx]
    player_cell = observation["grid"][centre, centre]
    return (
        int(player[PlayerFeature.ZONE]),
        int(player[PlayerFeature.FLOOR]),
        int(player[PlayerFeature.X]),
        int(player[PlayerFeature.Y]),
        int(action),
        *(int(value) for value in target),
        *(int(player_cell[channel]) for channel in _PLAYER_CONTROL_CHANNELS),
        *(int(value) for value in observation["inventory"].reshape(-1)),
    )


def apply_action_contract[Array: np.ndarray](
    observation: dict[str, Array], contract: str
) -> dict[str, Array]:
    """Apply a stateless contract without mutating authoritative live telemetry."""
    if contract not in ACTION_CONTRACTS:
        raise ValueError(f"Unknown action contract: {contract}")
    if contract in {"bounded-bard-v1", "bounded-bard-navigation-v1"}:
        scope = observation.get("bounded_loadout")
        if scope is None or np.any(scope[..., 0] != 1):
            raise ValueError(f"{contract} requires bounded-loadout-v1 telemetry")
        if np.any(scope[..., 1] != 1):
            raise UnsupportedLoadoutError(
                f"{contract} encountered equipment outside dagger/basic shovel/bombs; "
                "invalidate this experiment, do not score it as a gameplay failure"
            )
        return apply_action_contract(observation, "dagger-controls-v1")
    if contract == "dagger-controls-v1":
        controls = observation.get("equipment_controls")
        if controls is None or np.any(controls[..., 0] != 1):
            raise ValueError("dagger-controls-v1 requires equipment-controls-v1 telemetry")
        result = dict(observation)
        inventory = observation["inventory"].copy()
        supported = controls[..., 1].astype(bool)
        armed = controls[..., 2].astype(bool)
        # Versioned policy projection: this column remains toggle state for all
        # other equipment. Legacy contracts see the original inventory unchanged.
        inventory[..., 0, 7] = np.where(supported, armed, inventory[..., 0, 7])
        mask = observation["action_mask"].copy()
        # THROW is shared with equipment toggles/conversions. Only suppress a
        # repeat for the bounded loadout (weapon, shovel, bombs); keep it available
        # when any additional equipment could give that button another effect.
        extra_equipment = np.any(inventory[..., [1, 2, 4, 5, 7, 8, 9, 10, 11, 12], 0] != 0, axis=-1)
        mask[..., int(Action.THROW)] = np.where(
            armed & ~extra_equipment, 0, mask[..., int(Action.THROW)]
        )
        result["inventory"] = inventory
        result["action_mask"] = mask
        return result
    if contract in {"current", "known-invalid-wall-v1", "map-navigation-prior-v1"}:
        return observation
    result = dict(observation)
    mask = observation["action_mask"].copy()
    mask[..., int(Action.WAIT)] = 0
    result["action_mask"] = mask
    return result


@dataclass(slots=True)
class ActionContractMemory:
    """Maintain independent, episode-local action knowledge for worker slots."""

    contract: str
    slots: int
    _blocked: list[set[tuple[int, ...]]] = field(init=False, repr=False)
    _hazards: list[set[tuple[int, int]]] = field(init=False, repr=False)
    _hazard_levels: list[tuple[int, int] | None] = field(init=False, repr=False)
    _navigation: ActionContractMemory | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.contract not in ACTION_CONTRACTS:
            raise ValueError(f"Unknown action contract: {self.contract}")
        if self.slots <= 0:
            raise ValueError("slots must be positive")
        self._blocked = [set() for _ in range(self.slots)]
        self._hazards = [set() for _ in range(self.slots)]
        self._hazard_levels = [None for _ in range(self.slots)]
        if self.contract == "bounded-bard-navigation-v1":
            self._navigation = ActionContractMemory("dagger-controls-v1", self.slots)

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.slots:
            raise IndexError(f"Action-contract slot {slot} is outside capacity {self.slots}")

    def clear(self, slot: int) -> None:
        self._check_slot(slot)
        if self._navigation is not None:
            self._navigation.clear(slot)
        self._blocked[slot].clear()
        self._hazards[slot].clear()
        self._hazard_levels[slot] = None

    def reset_slot(self, slot: int, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.clear(slot)
        return self.apply_slot(slot, observation)

    def reset_batch(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if int(observation["action_mask"].shape[0]) != self.slots:
            raise ValueError("Batched action-mask capacity does not match contract slots")
        for slot in range(self.slots):
            self.clear(slot)
        return self.apply_batch(observation)

    def masked_directions(self, slot: int, observation: dict[str, np.ndarray]) -> tuple[int, ...]:
        self._check_slot(slot)
        if self._navigation is not None:
            return self._navigation.masked_directions(
                slot, apply_action_contract(observation, self.contract)
            )
        if self._armed(observation):
            return ()
        if self.contract not in _WALL_MEMORY_CONTRACTS:
            return ()
        base_mask = observation["action_mask"]
        return tuple(
            int(action)
            for action in DIRECTION_DELTAS
            if bool(base_mask[int(action)])
            and _state_signature(observation, action) in self._blocked[slot]
        )

    def apply_slot(self, slot: int, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._check_slot(slot)
        if self._navigation is not None:
            return self._navigation.apply_slot(
                slot, apply_action_contract(observation, self.contract)
            )
        if self.contract == "dagger-controls-v1":
            observation = apply_action_contract(observation, self.contract)
            if self._armed(observation):
                return observation
        if self.contract not in _WALL_MEMORY_CONTRACTS:
            return apply_action_contract(observation, self.contract)
        self._update_hazards(slot, observation)
        wall_masked = self.masked_directions(slot, observation)
        if self.contract in {"map-navigation-prior-v1", "dagger-controls-v1"}:
            navigation_preferred = _navigation_preferred_directions(observation, wall_masked)
        elif self.contract == "map-navigation-prior-v2":
            navigation_preferred = _navigation_preferred_directions_v2(
                observation, wall_masked, self._hazards[slot]
            )
        else:
            navigation_preferred = ()
        navigation_masked = tuple(
            int(action)
            for action in DIRECTION_DELTAS
            if bool(observation["action_mask"][int(action)])
            and int(action) not in wall_masked
            and not _direction_targets_wall(observation, int(action))
            and navigation_preferred
            and int(action) not in navigation_preferred
        )
        if not wall_masked and not navigation_masked:
            return observation
        result = dict(observation)
        action_mask = observation["action_mask"].copy()
        action_mask[list((*wall_masked, *navigation_masked))] = 0
        if navigation_preferred:
            action_mask[int(Action.WAIT)] = 0
        result["action_mask"] = action_mask
        return result

    def apply_batch(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._navigation is not None:
            return self._navigation.apply_batch(apply_action_contract(observation, self.contract))
        if self.contract == "dagger-controls-v1":
            if int(observation["action_mask"].shape[0]) != self.slots:
                raise ValueError("Batched action-mask capacity does not match contract slots")
            rows = [
                self.apply_slot(slot, {k: v[slot] for k, v in observation.items()})
                for slot in range(self.slots)
            ]
            result = dict(observation)
            for key in ("inventory", "action_mask"):
                result[key] = np.stack([row[key] for row in rows])
            return result
        if self.contract not in _WALL_MEMORY_CONTRACTS:
            return apply_action_contract(observation, self.contract)
        if int(observation["action_mask"].shape[0]) != self.slots:
            raise ValueError("Batched action-mask capacity does not match contract slots")
        effective_masks = [
            self.apply_slot(slot, {key: value[slot] for key, value in observation.items()})[
                "action_mask"
            ]
            for slot in range(self.slots)
        ]
        if all(
            np.array_equal(effective, observation["action_mask"][slot])
            for slot, effective in enumerate(effective_masks)
        ):
            return observation
        result = dict(observation)
        result["action_mask"] = np.stack(effective_masks)
        return result

    def observe(
        self,
        slot: int,
        before: dict[str, np.ndarray],
        action: int,
        after: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """Learn only authoritative wall no-ops and report the next effective mask."""
        self._check_slot(slot)
        if self._navigation is not None:
            result = self._navigation.observe(
                slot,
                apply_action_contract(before, self.contract),
                action,
                apply_action_contract(after, self.contract),
                info,
            )
            return {**result, "name": self.contract}
        newly_learned = False
        if self.contract == "bounded-bard-v1":
            apply_action_contract(before, self.contract)
            apply_action_contract(after, self.contract)
        category = str((info.get("action_outcome") or {}).get("category", ""))
        selected = Action(int(action))
        if (
            self.contract in _WALL_MEMORY_CONTRACTS
            and not self._armed(before)
            and selected in DIRECTION_DELTAS
            and category == "wall_attempt"
        ):
            signature = _state_signature(before, selected)
            newly_learned = signature not in self._blocked[slot]
            self._blocked[slot].add(signature)
        masked = self.masked_directions(slot, after)
        self._update_hazards(slot, after)
        if self._armed(after):
            navigation_preferred = ()
        elif self.contract in {"map-navigation-prior-v1", "dagger-controls-v1"}:
            navigation_preferred = _navigation_preferred_directions(after, masked)
        elif self.contract == "map-navigation-prior-v2":
            navigation_preferred = _navigation_preferred_directions_v2(
                after, masked, self._hazards[slot]
            )
        else:
            navigation_preferred = ()
        navigation_masked = (
            tuple(
                int(candidate)
                for candidate in DIRECTION_DELTAS
                if bool(after["action_mask"][int(candidate)])
                and int(candidate) not in masked
                and not _direction_targets_wall(after, int(candidate))
                and int(candidate) not in navigation_preferred
            )
            if navigation_preferred
            else ()
        )
        return {
            "name": self.contract,
            "newly_learned_invalid_wall": newly_learned,
            "masked_directions": list(masked),
            "masked_direction_count": len(masked),
            "effective_masked_direction_count": len(masked) + len(navigation_masked),
            "remembered_wall_states": len(self._blocked[slot]),
            "remembered_hazards": len(self._hazards[slot]),
            "navigation_prior_active": bool(navigation_preferred),
            "navigation_preferred_directions": list(navigation_preferred),
            "navigation_masked_directions": list(navigation_masked),
        }

    def _armed(self, observation: dict[str, np.ndarray]) -> bool:
        if self.contract != "dagger-controls-v1":
            return False
        controls = observation.get("equipment_controls")
        if controls is None or int(controls[0]) != 1:
            raise ValueError("dagger-controls-v1 requires equipment-controls-v1 telemetry")
        return bool(controls[1] and controls[2])

    def _update_hazards(self, slot: int, observation: dict[str, np.ndarray]) -> None:
        if self.contract != "map-navigation-prior-v2":
            return
        grid = observation["grid"]
        player = observation["player"]
        level = (
            int(player[PlayerFeature.ZONE]),
            int(player[PlayerFeature.FLOOR]),
        )
        if self._hazard_levels[slot] != level:
            self._hazards[slot].clear()
            self._hazard_levels[slot] = level
        px = int(player[PlayerFeature.X])
        py = int(player[PlayerFeature.Y])
        centre = GRID_SIZE // 2
        # VISIBILITY=1 is remembered/revealed terrain, not current line of sight.
        # Its dynamic channels are intentionally empty and must not erase a trap
        # that was previously observed at that coordinate.
        for row, column in np.argwhere(grid[..., GridChannel.VISIBILITY] == 2):
            position = (px + int(column) - centre, py + int(row) - centre)
            if int(grid[row, column, GridChannel.TRAP]) > 0:
                self._hazards[slot].add(position)
            else:
                self._hazards[slot].discard(position)


def _visible_enemy(observation: dict[str, np.ndarray]) -> bool:
    if int(observation["player"][PlayerFeature.VISIBLE_ENEMIES]) > 0:
        return True
    actors = observation["grid"][..., GridChannel.ACTOR_CLASS]
    return bool(np.any(actors > int(ActorKind.PLAYER)))


def _direction_targets_wall(observation: dict[str, np.ndarray], action_value: int) -> bool:
    action = Action(action_value)
    dx, dy = DIRECTION_DELTAS[action]
    return bool(
        int(
            observation["grid"][
                GRID_SIZE // 2 + dy,
                GRID_SIZE // 2 + dx,
                GridChannel.TERRAIN_CLASS,
            ]
        )
        == int(Terrain.WALL)
    )


def _stairs_first_steps(observation: dict[str, np.ndarray]) -> tuple[int, ...]:
    memory = observation["map_memory"]
    centre = memory.shape[0] // 2
    terrain = memory[..., MapChannel.TERRAIN_CLASS]
    targets = {
        tuple(int(value) for value in cell) for cell in np.argwhere(terrain == Terrain.STAIRS)
    }
    if not targets or (centre, centre) in targets:
        return ()
    queue: list[tuple[int, int, int | None, int]] = [(centre, centre, None, 0)]
    visited = {(centre, centre)}
    found_distance: int | None = None
    found: set[int] = set()
    head = 0
    while head < len(queue):
        row, column, first, distance = queue[head]
        head += 1
        if found_distance is not None and distance > found_distance:
            break
        if (row, column) in targets:
            found_distance = distance
            if first is not None:
                found.add(first)
            continue
        for action, (dx, dy) in DIRECTION_DELTAS.items():
            next_row, next_column = row + dy, column + dx
            if not (0 <= next_row < terrain.shape[0] and 0 <= next_column < terrain.shape[1]):
                continue
            position = (next_row, next_column)
            if position in visited:
                continue
            if int(terrain[position]) not in {int(Terrain.FLOOR), int(Terrain.STAIRS)}:
                continue
            visited.add(position)
            queue.append(
                (
                    next_row,
                    next_column,
                    int(action) if first is None else first,
                    distance + 1,
                )
            )
    return tuple(sorted(found))


def _navigation_preferred_directions(
    observation: dict[str, np.ndarray], wall_masked: tuple[int, ...]
) -> tuple[int, ...]:
    if _visible_enemy(observation):
        return ()
    base_mask = observation["action_mask"]
    candidates = tuple(
        int(action)
        for action in DIRECTION_DELTAS
        if bool(base_mask[int(action)]) and int(action) not in wall_masked
    )
    if not candidates:
        return ()
    stairs_steps = tuple(
        action for action in _stairs_first_steps(observation) if action in candidates
    )
    if stairs_steps:
        return stairs_steps
    memory = observation["map_memory"]
    centre = memory.shape[0] // 2

    def priority(action_value: int) -> tuple[int, int]:
        action = Action(action_value)
        dx, dy = DIRECTION_DELTAS[action]
        row, column = centre + dy, centre + dx
        visits = int(memory[row, column, MapChannel.VISIT_COUNT])
        terrain = int(
            observation["grid"][
                GRID_SIZE // 2 + dy,
                GRID_SIZE // 2 + dx,
                GridChannel.TERRAIN_CLASS,
            ]
        )
        # Prefer an observed traversable cell over a wall even when the wall has
        # never been attempted. Wall directions stay legal so the policy can
        # still dig; they simply do not suppress known-open exploration moves.
        return int(terrain == int(Terrain.WALL)), visits

    best = min(priority(action) for action in candidates)
    return tuple(action for action in candidates if priority(action) == best)


def _local_navigation_threat(observation: dict[str, np.ndarray]) -> bool:
    """Interrupt strategic routing only for a policy-relevant nearby threat."""
    if int(observation["player"][PlayerFeature.FLOOR]) >= 4:
        return True
    grid = observation["grid"]
    centre = GRID_SIZE // 2
    actors = grid[..., GridChannel.ACTOR_CLASS]
    for row, column in np.argwhere(actors > int(ActorKind.PLAYER)):
        distance = abs(int(row) - centre) + abs(int(column) - centre)
        actor = int(actors[row, column])
        if distance <= 2 or actor in {int(ActorKind.DRAGON), int(ActorKind.BOSS)}:
            return True
        if (
            int(grid[row, column, GridChannel.TELL_ANIMATION_DS]) > 0
            or int(grid[row, column, GridChannel.EXPLOSIVE]) > 0
        ):
            return True
    return False


def _planned_map_steps(
    observation: dict[str, np.ndarray],
    hazards: set[tuple[int, int]],
) -> tuple[int, ...]:
    memory = observation["map_memory"]
    terrain = memory[..., MapChannel.TERRAIN_CLASS]
    centre = terrain.shape[0] // 2
    player = observation["player"]
    px = int(player[PlayerFeature.X])
    py = int(player[PlayerFeature.Y])

    def hazardous(row: int, column: int) -> bool:
        return (px + column - centre, py + row - centre) in hazards

    stairs = {
        (int(row), int(column))
        for row, column in np.argwhere(terrain == int(Terrain.STAIRS))
        if not hazardous(int(row), int(column))
    }
    frontiers: set[tuple[int, int]] = set()
    if not stairs:
        for row, column in np.argwhere(terrain == int(Terrain.FLOOR)):
            row_value, column_value = int(row), int(column)
            if hazardous(row_value, column_value):
                continue
            if any(
                0 <= row_value + dy < terrain.shape[0]
                and 0 <= column_value + dx < terrain.shape[1]
                and int(terrain[row_value + dy, column_value + dx]) == int(Terrain.UNKNOWN)
                for dx, dy in DIRECTION_DELTAS.values()
            ):
                frontiers.add((row_value, column_value))
    goals = stairs or frontiers
    if not goals:
        return ()

    queue: list[tuple[int, int, int | None, int]] = [(centre, centre, None, 0)]
    visited = {(centre, centre)}
    found_distance: int | None = None
    found: set[int] = set()
    head = 0
    while head < len(queue):
        row, column, first, distance = queue[head]
        head += 1
        if found_distance is not None and distance > found_distance:
            break
        if (row, column) in goals and distance > 0:
            found_distance = distance
            if first is not None:
                found.add(first)
            continue
        for action, (dx, dy) in DIRECTION_DELTAS.items():
            next_row, next_column = row + dy, column + dx
            if not (0 <= next_row < terrain.shape[0] and 0 <= next_column < terrain.shape[1]):
                continue
            position = next_row, next_column
            if position in visited or hazardous(next_row, next_column):
                continue
            if int(terrain[position]) not in {int(Terrain.FLOOR), int(Terrain.STAIRS)}:
                continue
            visited.add(position)
            queue.append(
                (
                    next_row,
                    next_column,
                    int(action) if first is None else first,
                    distance + 1,
                )
            )
    return tuple(sorted(found))


def _navigation_preferred_directions_v2(
    observation: dict[str, np.ndarray],
    wall_masked: tuple[int, ...],
    hazards: set[tuple[int, int]],
) -> tuple[int, ...]:
    if _local_navigation_threat(observation):
        return ()
    candidates = tuple(
        int(action)
        for action in DIRECTION_DELTAS
        if bool(observation["action_mask"][int(action)]) and int(action) not in wall_masked
    )
    if not candidates:
        return ()
    planned = tuple(
        action for action in _planned_map_steps(observation, hazards) if action in candidates
    )
    if planned:
        return planned
    # Keep the v1 local fallback when no complete route exists yet. It permits
    # digging, but never lets an untested wall suppress a known-open direction.
    return _navigation_preferred_directions(observation, wall_masked)
