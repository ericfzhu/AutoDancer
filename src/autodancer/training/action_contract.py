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
    "current",
    "legacy-no-wait",
    "known-invalid-wall-v1",
    "map-navigation-prior-v1",
)
_WALL_MEMORY_CONTRACTS = {"known-invalid-wall-v1", "map-navigation-prior-v1"}

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


def _state_signature(
    observation: dict[str, np.ndarray], action: Action
) -> tuple[int, ...]:
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

    def __post_init__(self) -> None:
        if self.contract not in ACTION_CONTRACTS:
            raise ValueError(f"Unknown action contract: {self.contract}")
        if self.slots <= 0:
            raise ValueError("slots must be positive")
        self._blocked = [set() for _ in range(self.slots)]

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.slots:
            raise IndexError(f"Action-contract slot {slot} is outside capacity {self.slots}")

    def clear(self, slot: int) -> None:
        self._check_slot(slot)
        self._blocked[slot].clear()

    def reset_slot(
        self, slot: int, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        self.clear(slot)
        return self.apply_slot(slot, observation)

    def reset_batch(
        self, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if int(observation["action_mask"].shape[0]) != self.slots:
            raise ValueError("Batched action-mask capacity does not match contract slots")
        for slot in range(self.slots):
            self._blocked[slot].clear()
        return self.apply_batch(observation)

    def masked_directions(
        self, slot: int, observation: dict[str, np.ndarray]
    ) -> tuple[int, ...]:
        self._check_slot(slot)
        if self.contract not in _WALL_MEMORY_CONTRACTS:
            return ()
        base_mask = observation["action_mask"]
        return tuple(
            int(action)
            for action in DIRECTION_DELTAS
            if bool(base_mask[int(action)])
            and _state_signature(observation, action) in self._blocked[slot]
        )

    def apply_slot(
        self, slot: int, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        self._check_slot(slot)
        if self.contract not in _WALL_MEMORY_CONTRACTS:
            return apply_action_contract(observation, self.contract)
        wall_masked = self.masked_directions(slot, observation)
        navigation_preferred = (
            _navigation_preferred_directions(observation, wall_masked)
            if self.contract == "map-navigation-prior-v1"
            else ()
        )
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

    def apply_batch(
        self, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if self.contract not in _WALL_MEMORY_CONTRACTS:
            return apply_action_contract(observation, self.contract)
        if int(observation["action_mask"].shape[0]) != self.slots:
            raise ValueError("Batched action-mask capacity does not match contract slots")
        effective_masks = [
            self.apply_slot(
                slot, {key: value[slot] for key, value in observation.items()}
            )["action_mask"]
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
        newly_learned = False
        category = str((info.get("action_outcome") or {}).get("category", ""))
        selected = Action(int(action))
        if (
            self.contract in _WALL_MEMORY_CONTRACTS
            and selected in DIRECTION_DELTAS
            and category == "wall_attempt"
        ):
            signature = _state_signature(before, selected)
            newly_learned = signature not in self._blocked[slot]
            self._blocked[slot].add(signature)
        masked = self.masked_directions(slot, after)
        navigation_preferred = (
            _navigation_preferred_directions(after, masked)
            if self.contract == "map-navigation-prior-v1"
            else ()
        )
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
            "navigation_prior_active": bool(navigation_preferred),
            "navigation_preferred_directions": list(navigation_preferred),
            "navigation_masked_directions": list(navigation_masked),
        }


def _visible_enemy(observation: dict[str, np.ndarray]) -> bool:
    if int(observation["player"][PlayerFeature.VISIBLE_ENEMIES]) > 0:
        return True
    actors = observation["grid"][..., GridChannel.ACTOR_CLASS]
    return bool(np.any(actors > int(ActorKind.PLAYER)))


def _direction_targets_wall(
    observation: dict[str, np.ndarray], action_value: int
) -> bool:
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
        tuple(int(value) for value in cell)
        for cell in np.argwhere(terrain == Terrain.STAIRS)
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
            if not (
                0 <= next_row < terrain.shape[0]
                and 0 <= next_column < terrain.shape[1]
            ):
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
