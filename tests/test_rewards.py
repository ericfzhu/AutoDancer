from __future__ import annotations

import numpy as np
import pytest

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    PLAYER_FEATURES,
    GridChannel,
    PlayerFeature,
    Terrain,
)
from autodancer.rewards import RewardConfig, RewardTracker, load_reward_config


def observation(*, x: int = 0, y: int = 0) -> dict[str, np.ndarray]:
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16)
    grid[GRID_SIZE // 2, GRID_SIZE // 2, GridChannel.VISIBILITY] = 2
    player = np.zeros(PLAYER_FEATURES, dtype=np.int32)
    player[PlayerFeature.X] = x
    player[PlayerFeature.Y] = y
    return {
        "grid": grid,
        "player": player,
        "inventory": np.zeros((INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16),
        "action_mask": np.ones(ACTION_COUNT, dtype=np.int8),
    }


def test_exploration_reward_is_novelty_bounded_and_not_repeatable() -> None:
    tracker = RewardTracker()
    initial = observation()
    tracker.reset(initial, {"zone": 1, "floor": 1})
    moved = observation(x=1)
    moved["grid"][10, 11, GridChannel.VISIBILITY] = 2

    reward, parts = tracker.score(
        moved,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    assert parts == {"new_position": 0.005, "new_tile": 0.002}
    assert reward == pytest.approx(0.007)

    repeated, parts = tracker.score(
        moved,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    assert repeated == pytest.approx(0)
    assert parts == {}


def test_combat_credit_is_deduplicated_and_floor_bounded() -> None:
    tracker = RewardTracker(RewardConfig(max_combat_reward_per_floor=0.05))
    value = observation()
    tracker.reset(value, {"zone": 1, "floor": 1})
    events = [
        {"kind": "enemy_damage", "amount": 10, "entity_id": 91},
        {"kind": "enemy_kill", "amount": 1, "entity_id": 91},
        {"kind": "enemy_kill", "amount": 1, "entity_id": 91},
        {"kind": "player_damage", "amount": 2, "entity_id": 91},
    ]
    reward, parts = tracker.score(
        value,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        events,
        terminated=False,
        truncated=False,
    )
    assert parts["enemy_damage"] == pytest.approx(0.05)
    assert "enemy_kill" not in parts
    assert parts["player_damage"] == pytest.approx(-0.3)
    assert reward == pytest.approx(-0.25)


def test_progress_inventory_and_terminal_rewards_dominate_shaping() -> None:
    tracker = RewardTracker()
    initial = observation()
    tracker.reset(initial, {"zone": 1, "floor": 1})
    progressed = observation()
    progressed["inventory"][0, 1] = 1234

    floor_reward, parts = tracker.score(
        progressed,
        {"zone": 1, "floor": 2, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    assert parts["floor_complete"] == 5.0
    assert parts["new_item_type"] == 0.10
    assert parts["new_position"] == 0.005
    assert floor_reward == pytest.approx(5.106)

    victory_reward, parts = tracker.score(
        progressed,
        {"zone": 1, "floor": 2, "episode_status": "won"},
        [{"kind": "success", "amount": 1}],
        terminated=True,
        truncated=False,
    )
    assert parts["victory"] == 50.0
    assert victory_reward == pytest.approx(50.0)


def test_currency_reward_consumes_currency_event_not_item_event() -> None:
    tracker = RewardTracker(
        RewardConfig(currency=0.1, new_position=0, new_tile=0, stair_potential_max=0)
    )
    value = observation()
    tracker.reset(value, {"zone": 1, "floor": 1})
    reward, parts = tracker.score(
        value,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [
            {"kind": "currency_collected", "amount": 3},
            {"kind": "item_collected", "amount": 99},
        ],
        terminated=False,
        truncated=False,
    )
    assert reward == pytest.approx(0.3)
    assert parts == {"currency": pytest.approx(0.3)}


def test_reward_configuration_rejects_unknown_fields(tmp_path) -> None:
    path = tmp_path / "reward.json"
    path.write_text('{"turn": -0.01, "survival": 9}', encoding="utf-8")
    with pytest.raises(ValueError, match="survival"):
        load_reward_config(path)

    path.write_text('{"turn": -0.01}', encoding="utf-8")
    assert load_reward_config(path).turn == -0.01


def test_v4_arm_configs_differ_only_in_stair_potential() -> None:
    arm_a = load_reward_config("configs/reward-v4a.json")
    arm_b = load_reward_config("configs/reward-v4b.json")
    assert arm_a.stair_potential_max == 0.5
    assert arm_b.stair_potential_max == 1.0
    a_weights = arm_a.specification()["weights"]
    b_weights = arm_b.specification()["weights"]
    assert {key for key in a_weights if a_weights[key] != b_weights[key]} == {"stair_potential_max"}


def test_v2_config_restores_exact_legacy_metadata_and_stair_delta() -> None:
    config = load_reward_config("configs/reward-v2.json")
    assert config.specification() == {
        "version": 2,
        "weights": {
            "turn": -0.005,
            "new_position": 0.015,
            "revisit": -0.01,
            "new_tile": 0.001,
            "max_new_tiles_per_turn": 25,
            "enemy_damage": 0.025,
            "max_rewarded_damage_per_enemy": 16,
            "enemy_kill": 0.25,
            "player_damage": -0.15,
            "new_item_type": 0.15,
            "currency": 0.002,
            "max_currency_per_turn": 25,
            "container_opened": 0.05,
            "stairs_discovered": 0.5,
            "stair_progress": 0.05,
            "max_stair_distance_delta": 4,
            "floor_complete": 5.0,
            "zone_complete": 10.0,
            "victory": 50.0,
            "death": -2.0,
            "aborted": -1.0,
        },
    }
    tracker = RewardTracker(config)
    tracker.reset(observation(), {"zone": 1, "floor": 1})
    discovered = observation()
    discovered["grid"][10, 12, GridChannel.TERRAIN_CLASS] = Terrain.STAIRS
    discovered["grid"][10, 12, GridChannel.VISIBILITY] = 1
    _, discovery = tracker.score(
        discovered,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    closer = observation(x=1)
    closer["grid"][10, 11, GridChannel.TERRAIN_CLASS] = Terrain.STAIRS
    closer["grid"][10, 11, GridChannel.VISIBILITY] = 1
    _, progress = tracker.score(
        closer,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    assert discovery["stairs_discovered"] == 0.5
    assert "stair_potential" not in discovery
    assert progress["stair_progress"] == 0.05


def test_v2_zone_transition_does_not_add_floor_completion() -> None:
    tracker = RewardTracker(load_reward_config("configs/reward-v2.json"))
    tracker.reset(observation(), {"zone": 1, "floor": 3})
    _, parts = tracker.score(
        observation(),
        {"zone": 2, "floor": 1, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    assert parts["zone_complete"] == 10.0
    assert "floor_complete" not in parts


def test_stair_potential_rewards_progress_and_charges_exactly_for_reversal() -> None:
    config = RewardConfig(
        turn=0,
        new_position=0,
        revisit=0,
        new_tile=0,
    )
    tracker = RewardTracker(config)
    initial = observation(x=0)
    tracker.reset(initial, {"zone": 1, "floor": 1})

    discovered = observation(x=0)
    discovered["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 2, GridChannel.TERRAIN_CLASS] = (
        Terrain.STAIRS
    )
    discovered["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 2, GridChannel.VISIBILITY] = 1
    reward, parts = tracker.score(
        discovered,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    assert reward == pytest.approx(0.99 * 0.45)
    assert parts["stair_potential"] == pytest.approx(0.99 * 0.45)

    closer = observation(x=1)
    closer["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = Terrain.STAIRS
    closer["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.VISIBILITY] = 1
    toward, toward_parts = tracker.score(
        closer,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    away, away_parts = tracker.score(
        discovered,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    assert toward_parts["stair_potential"] == pytest.approx(0.99 * 0.475 - 0.45)
    assert away_parts["stair_potential"] == pytest.approx(0.99 * 0.45 - 0.475)
    assert toward + config.discount * away == pytest.approx(-0.45 + config.discount**2 * 0.45)


def test_floor_transition_resets_stair_potential_without_cross_floor_credit() -> None:
    tracker = RewardTracker(RewardConfig(turn=0, new_position=0, revisit=0, new_tile=0))
    initial = observation()
    initial["grid"][10, 11, GridChannel.TERRAIN_CLASS] = Terrain.STAIRS
    tracker.reset(initial, {"zone": 1, "floor": 1})
    next_floor = observation(x=20, y=20)
    reward, parts = tracker.score(
        next_floor,
        {"zone": 1, "floor": 2, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    assert reward == pytest.approx(5.0 - 0.475)
    assert parts["stair_potential"] == pytest.approx(-0.475)


def test_time_limit_retains_stair_potential_and_is_not_an_abort() -> None:
    config = RewardConfig(turn=0, new_position=0, revisit=0, new_tile=0)
    tracker = RewardTracker(config)
    state = observation()
    state["grid"][10, 12, GridChannel.TERRAIN_CLASS] = Terrain.STAIRS
    tracker.reset(state, {"zone": 1, "floor": 1})
    reward, parts = tracker.score(
        state,
        {"zone": 1, "floor": 1, "episode_status": "time_limit"},
        [],
        terminated=False,
        truncated=True,
    )
    assert reward == pytest.approx(config.discount * 0.45 - 0.45)
    assert "aborted" not in parts


def test_shaping_budgets_reset_on_floor_transition() -> None:
    config = RewardConfig(
        new_position=0.1,
        new_tile=0,
        max_exploration_reward_per_floor=0.1,
        enemy_kill=0.1,
        max_combat_reward_per_floor=0.1,
        new_item_type=0.1,
        max_item_reward_per_floor=0.1,
    )
    tracker = RewardTracker(config)
    tracker.reset(observation(), {"zone": 1, "floor": 1})
    value = observation(x=1)
    value["inventory"][0, 1] = 11
    _, first = tracker.score(
        value,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [{"kind": "enemy_kill", "entity_id": 1}],
        terminated=False,
        truncated=False,
    )
    value2 = observation(x=2)
    value2["inventory"][0, 1] = 11
    value2["inventory"][1, 1] = 12
    _, capped = tracker.score(
        value2,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [{"kind": "enemy_kill", "entity_id": 2}],
        terminated=False,
        truncated=False,
    )
    _, reset = tracker.score(
        observation(x=3),
        {"zone": 1, "floor": 2, "episode_status": "running"},
        [{"kind": "enemy_kill", "entity_id": 3}],
        terminated=False,
        truncated=False,
    )
    assert first["new_position"] == first["enemy_kill"] == first["new_item_type"] == 0.1
    assert "new_position" not in capped
    assert "enemy_kill" not in capped
    assert "new_item_type" not in capped
    assert reset["new_position"] == reset["enemy_kill"] == 0.1


def test_reward_components_split_task_and_shaping_returns() -> None:
    extrinsic, shaping = RewardTracker.split_components(
        {"floor_complete": 5.0, "death": -1.0, "new_position": 0.005}
    )
    assert extrinsic == pytest.approx(4.0)
    assert shaping == pytest.approx(0.005)
