from __future__ import annotations

import numpy as np

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
    PlayerFeature,
    Terrain,
)
from autodancer.training.action_contract import ActionContractMemory, apply_action_contract


def _observation(*, x: int = 4, y: int = 7) -> dict[str, np.ndarray]:
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16)
    grid[..., GridChannel.TERRAIN_CLASS] = int(Terrain.FLOOR)
    player = np.zeros(PLAYER_FEATURES, dtype=np.int32)
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 1
    player[PlayerFeature.X] = x
    player[PlayerFeature.Y] = y
    return {
        "grid": grid,
        "player": player,
        "inventory": np.zeros(
            (INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16
        ),
        "action_mask": np.ones(ACTION_COUNT, dtype=np.int8),
    }


def test_legacy_contract_masks_wait_without_mutating_live_observation() -> None:
    live = {"action_mask": np.ones((2, ACTION_COUNT), dtype=np.int8)}
    legacy = apply_action_contract(live, "legacy-no-wait")
    assert np.all(live["action_mask"][:, int(Action.WAIT)] == 1)
    assert np.all(legacy["action_mask"][:, int(Action.WAIT)] == 0)
    assert np.all(legacy["action_mask"][:, :4] == 1)


def test_current_contract_returns_live_observation_unchanged() -> None:
    live = {"action_mask": np.ones(ACTION_COUNT, dtype=np.int8)}
    assert apply_action_contract(live, "current") is live


def test_known_invalid_wall_masks_only_after_authoritative_no_op() -> None:
    memory = ActionContractMemory("known-invalid-wall-v1", 1)
    live = _observation()
    live["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = (
        Terrain.WALL
    )
    effective = memory.reset_slot(0, live)
    assert effective["action_mask"][Action.RIGHT] == 1

    diagnostic = memory.observe(
        0,
        effective,
        Action.RIGHT,
        live,
        {"action_outcome": {"category": "wall_attempt"}},
    )
    effective = memory.apply_slot(0, live)
    assert diagnostic["newly_learned_invalid_wall"]
    assert diagnostic["masked_directions"] == [int(Action.RIGHT)]
    assert effective["action_mask"][Action.RIGHT] == 0
    assert np.all(live["action_mask"] == 1)
    assert np.all(effective["action_mask"][[Action.UP, Action.DOWN, Action.LEFT]] == 1)


def test_known_invalid_wall_reopens_when_target_actor_or_inventory_changes() -> None:
    memory = ActionContractMemory("known-invalid-wall-v1", 1)
    live = _observation()
    target = (GRID_SIZE // 2, GRID_SIZE // 2 + 1)
    live["grid"][*target, GridChannel.TERRAIN_CLASS] = Terrain.WALL
    memory.reset_slot(0, live)
    memory.observe(
        0,
        live,
        Action.RIGHT,
        live,
        {"action_outcome": {"category": "wall_attempt"}},
    )

    enemy_changed = {key: value.copy() for key, value in live.items()}
    enemy_changed["grid"][*target, GridChannel.ACTOR_CLASS] = ActorKind.SKELETON
    assert memory.apply_slot(0, enemy_changed)["action_mask"][Action.RIGHT] == 1

    inventory_changed = {key: value.copy() for key, value in live.items()}
    inventory_changed["inventory"][0, 1] = 123
    assert memory.apply_slot(0, inventory_changed)["action_mask"][Action.RIGHT] == 1


def test_known_invalid_wall_never_learns_dig_or_combat() -> None:
    memory = ActionContractMemory("known-invalid-wall-v1", 1)
    live = _observation()
    live["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = (
        Terrain.WALL
    )
    memory.reset_slot(0, live)
    for category in ("dig", "combat", "combat_attempt"):
        diagnostic = memory.observe(
            0,
            live,
            Action.RIGHT,
            live,
            {"action_outcome": {"category": category}},
        )
        assert not diagnostic["newly_learned_invalid_wall"]
        assert memory.apply_slot(0, live)["action_mask"][Action.RIGHT] == 1


def test_known_invalid_wall_memory_is_slot_local_and_clears_on_reset() -> None:
    memory = ActionContractMemory("known-invalid-wall-v1", 2)
    live = _observation()
    live["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = (
        Terrain.WALL
    )
    batch = {key: np.stack([value, value]) for key, value in live.items()}
    memory.reset_batch(batch)
    memory.observe(
        0,
        live,
        Action.RIGHT,
        live,
        {"action_outcome": {"category": "wall_attempt"}},
    )
    effective = memory.apply_batch(batch)
    assert effective["action_mask"][0, Action.RIGHT] == 0
    assert effective["action_mask"][1, Action.RIGHT] == 1
    assert memory.reset_slot(0, live)["action_mask"][Action.RIGHT] == 1
