"""Policy-side action contracts, including episode-local invalid-action memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autodancer.constants import (
    DIRECTION_DELTAS,
    GRID_SIZE,
    Action,
    GridChannel,
    PlayerFeature,
)

ACTION_CONTRACTS = ("current", "legacy-no-wait", "known-invalid-wall-v1")

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
    if contract in {"current", "known-invalid-wall-v1"}:
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
        if self.contract != "known-invalid-wall-v1":
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
        if self.contract != "known-invalid-wall-v1":
            return apply_action_contract(observation, self.contract)
        masked = self.masked_directions(slot, observation)
        if not masked:
            return observation
        result = dict(observation)
        action_mask = observation["action_mask"].copy()
        action_mask[list(masked)] = 0
        result["action_mask"] = action_mask
        return result

    def apply_batch(
        self, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if self.contract != "known-invalid-wall-v1":
            return apply_action_contract(observation, self.contract)
        if int(observation["action_mask"].shape[0]) != self.slots:
            raise ValueError("Batched action-mask capacity does not match contract slots")
        masked_by_slot = [
            self.masked_directions(
                slot,
                {key: value[slot] for key, value in observation.items()},
            )
            for slot in range(self.slots)
        ]
        if not any(masked_by_slot):
            return observation
        result = dict(observation)
        action_mask = observation["action_mask"].copy()
        for slot, masked in enumerate(masked_by_slot):
            if masked:
                action_mask[slot, list(masked)] = 0
        result["action_mask"] = action_mask
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
            self.contract == "known-invalid-wall-v1"
            and selected in DIRECTION_DELTAS
            and category == "wall_attempt"
        ):
            signature = _state_signature(before, selected)
            newly_learned = signature not in self._blocked[slot]
            self._blocked[slot].add(signature)
        masked = self.masked_directions(slot, after)
        return {
            "name": self.contract,
            "newly_learned_invalid_wall": newly_learned,
            "masked_directions": list(masked),
            "masked_direction_count": len(masked),
            "remembered_wall_states": len(self._blocked[slot]),
        }
