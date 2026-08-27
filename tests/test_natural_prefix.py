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
    ActorKind,
    BossType,
    GridChannel,
    PlayerFeature,
)
from autodancer.training.natural_prefix import (
    DeathMetalPhaseTracker,
    NaturalPrefixConfig,
)


def boss_observation(*, health: int, actor_type: int) -> dict[str, np.ndarray]:
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16)
    grid[10, 11, GridChannel.ACTOR_CLASS] = int(ActorKind.BOSS)
    grid[10, 11, GridChannel.ACTOR_TYPE] = actor_type
    grid[10, 11, GridChannel.HEALTH] = health
    grid[10, 11, GridChannel.MAX_HEALTH] = 9
    grid[10, 11, GridChannel.VISIBILITY] = 2
    player = np.zeros(PLAYER_FEATURES, dtype=np.int32)
    player[PlayerFeature.TASK] = int(BossType.DEATH_METAL)
    return {
        "grid": grid,
        "map_memory": np.zeros((MAP_SIZE, MAP_SIZE, MAP_CHANNELS), dtype=np.int16),
        "player": player,
        "inventory": np.zeros((INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16),
        "action_mask": np.ones(ACTION_COUNT, dtype=np.int8),
    }


def damage(amount: int) -> dict:
    return {"raw_events": [{"kind": "enemy_damage", "amount": amount, "data": {"boss": True}}]}


def test_death_metal_phase4_requires_health_damage_and_four_entity_types() -> None:
    tracker = DeathMetalPhaseTracker(NaturalPrefixConfig(target_phase=4))
    tracker.observe(boss_observation(health=9, actor_type=101))
    tracker.observe(boss_observation(health=6, actor_type=102), damage(3))
    tracker.observe(boss_observation(health=4, actor_type=103), damage(2))
    tracker.observe(boss_observation(health=2, actor_type=104), damage(2))
    assert tracker.reached
    assert tracker.snapshot()["observed_actor_types"] == [101, 102, 103, 104]


def test_direct_health_mutation_cannot_satisfy_natural_handoff() -> None:
    tracker = DeathMetalPhaseTracker(NaturalPrefixConfig(target_phase=4))
    tracker.observe(boss_observation(health=9, actor_type=101))
    tracker.observe(boss_observation(health=1, actor_type=101))
    assert not tracker.reached
    assert tracker.snapshot()["boss_damage"] == 0


def test_phase_handoff_fails_closed_on_type_hash_collision() -> None:
    tracker = DeathMetalPhaseTracker(NaturalPrefixConfig(target_phase=3))
    tracker.observe(boss_observation(health=9, actor_type=101))
    tracker.observe(boss_observation(health=6, actor_type=101), damage(3))
    tracker.observe(boss_observation(health=4, actor_type=101), damage(2))
    assert not tracker.reached


def test_non_death_metal_observation_does_not_advance_tracker() -> None:
    observation = boss_observation(health=2, actor_type=104)
    observation["player"][PlayerFeature.TASK] = int(BossType.KING_CONGA)
    tracker = DeathMetalPhaseTracker(NaturalPrefixConfig(target_phase=4))
    tracker.observe(observation, damage(9))
    assert not tracker.reached
    assert tracker.snapshot()["observations_with_boss"] == 0
