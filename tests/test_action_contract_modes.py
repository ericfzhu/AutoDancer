from __future__ import annotations

import numpy as np

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    MAP_CHANNELS,
    MAP_SIZE,
    PLAYER_FEATURES,
    Action,
    ActorKind,
    GridChannel,
    MapChannel,
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
        "map_memory": np.zeros((MAP_SIZE, MAP_SIZE, MAP_CHANNELS), dtype=np.int16),
        "player": player,
        "inventory": np.zeros((INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16),
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
    live["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = Terrain.WALL
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
    live["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = Terrain.WALL
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
    live["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = Terrain.WALL
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


def test_map_navigation_prior_selects_least_visited_direction_and_masks_wait() -> None:
    memory = ActionContractMemory("map-navigation-prior-v1", 1)
    live = _observation()
    centre = MAP_SIZE // 2
    visits = {
        Action.UP: 0,
        Action.RIGHT: 2,
        Action.DOWN: 3,
        Action.LEFT: 4,
    }
    for action, count in visits.items():
        dx, dy = {
            Action.UP: (0, -1),
            Action.RIGHT: (1, 0),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
        }[action]
        live["map_memory"][centre + dy, centre + dx, MapChannel.TERRAIN_CLASS] = Terrain.FLOOR
        live["map_memory"][centre + dy, centre + dx, MapChannel.VISIT_COUNT] = count

    effective = memory.reset_slot(0, live)

    assert effective["action_mask"][Action.UP] == 1
    assert np.all(
        effective["action_mask"][[Action.RIGHT, Action.DOWN, Action.LEFT, Action.WAIT]] == 0
    )


def test_map_navigation_prior_disengages_when_enemy_is_visible() -> None:
    memory = ActionContractMemory("map-navigation-prior-v1", 1)
    live = _observation()
    live["player"][PlayerFeature.VISIBLE_ENEMIES] = 1

    effective = memory.reset_slot(0, live)

    assert np.all(effective["action_mask"] == live["action_mask"])


def test_map_navigation_prior_routes_toward_known_stairs() -> None:
    memory = ActionContractMemory("map-navigation-prior-v1", 1)
    live = _observation()
    centre = MAP_SIZE // 2
    live["map_memory"][..., MapChannel.TERRAIN_CLASS] = Terrain.FLOOR
    live["map_memory"][centre, centre + 2, MapChannel.TERRAIN_CLASS] = Terrain.STAIRS

    effective = memory.reset_slot(0, live)

    assert effective["action_mask"][Action.RIGHT] == 1
    assert np.all(effective["action_mask"][[Action.UP, Action.DOWN, Action.LEFT, Action.WAIT]] == 0)


def test_map_navigation_prior_never_suppresses_possible_dig() -> None:
    memory = ActionContractMemory("map-navigation-prior-v1", 1)
    live = _observation()
    centre = GRID_SIZE // 2
    live["grid"][centre, centre + 1, GridChannel.TERRAIN_CLASS] = Terrain.WALL

    effective = memory.reset_slot(0, live)

    assert effective["action_mask"][Action.RIGHT] == 1
    assert effective["action_mask"][Action.UP] == 1


def test_map_navigation_prior_applies_independently_to_batched_slots() -> None:
    memory = ActionContractMemory("map-navigation-prior-v1", 2)
    first = _observation()
    second = _observation()
    centre = MAP_SIZE // 2
    first["map_memory"][centre, centre + 1, MapChannel.VISIT_COUNT] = 3
    second["map_memory"][centre - 1, centre, MapChannel.VISIT_COUNT] = 3
    batch = {key: np.stack([first[key], second[key]]) for key in first}

    effective = memory.reset_batch(batch)

    assert effective["action_mask"][0, Action.RIGHT] == 0
    assert effective["action_mask"][1, Action.UP] == 0
    assert effective["action_mask"][0, Action.UP] == 1
    assert effective["action_mask"][1, Action.RIGHT] == 1


def test_navigation_v2_routes_to_frontier_despite_far_visible_enemy() -> None:
    memory = ActionContractMemory("map-navigation-prior-v2", 1)
    live = _observation()
    map_centre = MAP_SIZE // 2
    grid_centre = GRID_SIZE // 2
    live["map_memory"][..., MapChannel.TERRAIN_CLASS] = Terrain.WALL
    live["map_memory"][map_centre, map_centre : map_centre + 4, MapChannel.TERRAIN_CLASS] = (
        Terrain.FLOOR
    )
    live["map_memory"][map_centre - 1, map_centre, MapChannel.TERRAIN_CLASS] = Terrain.FLOOR
    live["map_memory"][map_centre - 1, map_centre, MapChannel.VISIT_COUNT] = 0
    live["map_memory"][map_centre, map_centre + 1, MapChannel.VISIT_COUNT] = 9
    live["map_memory"][map_centre, map_centre + 4, MapChannel.TERRAIN_CLASS] = Terrain.UNKNOWN
    live["grid"][grid_centre - 5, grid_centre, GridChannel.ACTOR_CLASS] = ActorKind.SKELETON
    live["player"][PlayerFeature.VISIBLE_ENEMIES] = 1

    effective = memory.reset_slot(0, live)

    assert effective["action_mask"][Action.RIGHT] == 1
    assert np.all(effective["action_mask"][[Action.UP, Action.DOWN, Action.LEFT, Action.WAIT]] == 0)


def test_navigation_v2_yields_to_nearby_enemy_and_boss_floor() -> None:
    memory = ActionContractMemory("map-navigation-prior-v2", 1)
    nearby = _observation()
    centre = GRID_SIZE // 2
    nearby["grid"][centre, centre + 2, GridChannel.ACTOR_CLASS] = ActorKind.SKELETON
    assert np.all(memory.reset_slot(0, nearby)["action_mask"] == 1)

    boss_floor = _observation()
    boss_floor["player"][PlayerFeature.FLOOR] = 4
    assert np.all(memory.reset_slot(0, boss_floor)["action_mask"] == 1)


def test_navigation_v2_remembers_visible_trap_and_routes_around_it() -> None:
    memory = ActionContractMemory("map-navigation-prior-v2", 1)
    live = _observation()
    map_centre = MAP_SIZE // 2
    grid_centre = GRID_SIZE // 2
    path = (
        (0, 0),
        (0, -1),
        (1, -1),
        (2, -1),
        (2, 0),
        (1, 0),
    )
    for dx, dy in path:
        live["map_memory"][map_centre + dy, map_centre + dx, MapChannel.TERRAIN_CLASS] = (
            Terrain.FLOOR
        )
    live["map_memory"][map_centre, map_centre + 2, MapChannel.TERRAIN_CLASS] = Terrain.STAIRS
    live["grid"][grid_centre, grid_centre + 1, GridChannel.VISIBILITY] = 2
    live["grid"][grid_centre, grid_centre + 1, GridChannel.TRAP] = 1

    effective = memory.reset_slot(0, live)
    diagnostic = memory.observe(
        0,
        effective,
        Action.UP,
        live,
        {"action_outcome": {"category": "move"}},
    )

    assert effective["action_mask"][Action.UP] == 1
    assert effective["action_mask"][Action.RIGHT] == 0
    assert diagnostic["remembered_hazards"] == 1


def test_navigation_v2_warned_hazard_is_slot_local_and_clears_on_reset() -> None:
    memory = ActionContractMemory("map-navigation-prior-v2", 2)
    trap = _observation()
    centre = GRID_SIZE // 2
    trap["grid"][centre, centre + 1, GridChannel.VISIBILITY] = 2
    trap["grid"][centre, centre + 1, GridChannel.TRAP] = 1
    memory.reset_slot(0, trap)
    diagnostic = memory.observe(
        0,
        trap,
        Action.UP,
        trap,
        {"action_outcome": {"category": "move"}},
    )
    assert diagnostic["remembered_hazards"] == 1

    safe = _observation(x=20, y=30)
    safe_diagnostic = memory.observe(
        1,
        safe,
        Action.UP,
        safe,
        {"action_outcome": {"category": "move"}},
    )
    assert safe_diagnostic["remembered_hazards"] == 0
    memory.reset_slot(0, safe)
    cleared = memory.observe(
        0,
        safe,
        Action.UP,
        safe,
        {"action_outcome": {"category": "move"}},
    )
    assert cleared["remembered_hazards"] == 0


def test_navigation_v2_batched_reset_clears_hazards_between_seed_waves() -> None:
    memory = ActionContractMemory("map-navigation-prior-v2", 1)
    trap = _observation()
    centre = GRID_SIZE // 2
    trap["grid"][centre, centre + 1, GridChannel.VISIBILITY] = 2
    trap["grid"][centre, centre + 1, GridChannel.TRAP] = 1
    memory.reset_slot(0, trap)
    assert (
        memory.observe(
            0,
            trap,
            Action.UP,
            trap,
            {"action_outcome": {"category": "move"}},
        )["remembered_hazards"]
        == 1
    )

    fresh = _observation(x=40, y=40)
    memory.reset_batch({key: np.expand_dims(value, 0) for key, value in fresh.items()})
    assert (
        memory.observe(
            0,
            fresh,
            Action.UP,
            fresh,
            {"action_outcome": {"category": "move"}},
        )["remembered_hazards"]
        == 0
    )


def test_navigation_v2_clears_hazards_on_natural_floor_transition() -> None:
    memory = ActionContractMemory("map-navigation-prior-v2", 1)
    trap = _observation(x=12, y=15)
    centre = GRID_SIZE // 2
    trap["grid"][centre, centre + 1, GridChannel.VISIBILITY] = 2
    trap["grid"][centre, centre + 1, GridChannel.TRAP] = 1
    memory.reset_slot(0, trap)
    assert (
        memory.observe(
            0,
            trap,
            Action.UP,
            trap,
            {"action_outcome": {"category": "move"}},
        )["remembered_hazards"]
        == 1
    )

    next_floor = _observation(x=12, y=15)
    next_floor["player"][PlayerFeature.FLOOR] = 2
    diagnostic = memory.observe(
        0,
        trap,
        Action.UP,
        next_floor,
        {"action_outcome": {"category": "floor_transition"}},
    )
    assert diagnostic["remembered_hazards"] == 0


def test_navigation_v2_revealed_offscreen_tile_does_not_erase_hazard() -> None:
    memory = ActionContractMemory("map-navigation-prior-v2", 1)
    visible_trap = _observation(x=12, y=15)
    centre = GRID_SIZE // 2
    visible_trap["grid"][centre, centre + 1, GridChannel.VISIBILITY] = 2
    visible_trap["grid"][centre, centre + 1, GridChannel.TRAP] = 1
    memory.reset_slot(0, visible_trap)

    remembered_only = _observation(x=12, y=15)
    remembered_only["grid"][centre, centre + 1, GridChannel.VISIBILITY] = 1
    diagnostic = memory.observe(
        0,
        visible_trap,
        Action.UP,
        remembered_only,
        {"action_outcome": {"category": "move"}},
    )
    assert diagnostic["remembered_hazards"] == 1
