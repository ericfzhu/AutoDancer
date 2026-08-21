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
    assert parts == {"turn": -0.005, "new_position": 0.015, "new_tile": 0.002}
    assert reward == pytest.approx(0.012)

    repeated, parts = tracker.score(
        moved,
        {"zone": 1, "floor": 1, "episode_status": "running"},
        [],
        terminated=False,
        truncated=False,
    )
    assert repeated == pytest.approx(-0.005)
    assert parts == {"turn": -0.005}


def test_combat_credit_is_deduplicated_and_damage_is_capped() -> None:
    tracker = RewardTracker(RewardConfig(max_rewarded_damage_per_enemy=3))
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
    assert parts["enemy_damage"] == pytest.approx(0.075)
    assert parts["enemy_kill"] == pytest.approx(0.25)
    assert parts["player_damage"] == pytest.approx(-0.3)
    assert reward == pytest.approx(0.02)


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
    assert parts["floor_complete"] == 3.0
    assert parts["new_item_type"] == 0.15
    assert parts["new_position"] == 0.015
    assert floor_reward == pytest.approx(3.161)

    victory_reward, parts = tracker.score(
        progressed,
        {"zone": 1, "floor": 2, "episode_status": "won"},
        [{"kind": "success", "amount": 1}],
        terminated=True,
        truncated=False,
    )
    assert parts["victory"] == 25.0
    assert victory_reward == pytest.approx(24.995)


def test_reward_configuration_rejects_unknown_fields(tmp_path) -> None:
    path = tmp_path / "reward.json"
    path.write_text('{"turn": -0.01, "survival": 9}', encoding="utf-8")
    with pytest.raises(ValueError, match="survival"):
        load_reward_config(path)

    path.write_text('{"turn": -0.01}', encoding="utf-8")
    assert load_reward_config(path).turn == -0.01
