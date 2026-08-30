from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

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
from autodancer.rewards import RewardConfig
from autodancer.training.natural_prefix import (
    DeathMetalPhaseTracker,
    NaturalPrefixConfig,
    guide_reward_config,
    natural_prefix_identity,
    natural_prefix_policy_sample,
    validate_guide_action_contract,
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


def damage(amount: int, actor_type: int = 0) -> dict:
    return {
        "raw_events": [
            {
                "kind": "enemy_damage",
                "amount": amount,
                "data": {"boss": True, "actor_type": actor_type},
            }
        ]
    }


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


def test_phase_tracker_retains_authoritative_damage_event_types_offscreen() -> None:
    tracker = DeathMetalPhaseTracker(NaturalPrefixConfig(target_phase=4))
    tracker.observe(boss_observation(health=9, actor_type=101))
    tracker.observe(boss_observation(health=6, actor_type=102), damage(3, 102))
    tracker.observe(boss_observation(health=4, actor_type=103), damage(2, 103))
    observation = boss_observation(health=2, actor_type=104)
    observation["grid"][..., GridChannel.VISIBILITY] = 0
    tracker.observe(observation, damage(2, 104))

    assert tracker.reached
    assert tracker.snapshot()["observed_actor_types"] == [101, 102, 103, 104]


def test_non_death_metal_observation_does_not_advance_tracker() -> None:
    observation = boss_observation(health=2, actor_type=104)
    observation["player"][PlayerFeature.TASK] = int(BossType.KING_CONGA)
    tracker = DeathMetalPhaseTracker(NaturalPrefixConfig(target_phase=4))
    tracker.observe(observation, damage(9))
    assert not tracker.reached
    assert tracker.snapshot()["observations_with_boss"] == 0


def test_natural_prefix_failure_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="turn and attempt limits"):
        NaturalPrefixConfig(max_failed_seeds_per_fragment=0)


def test_guide_action_contract_must_match_checkpoint() -> None:
    payload = {"checkpoint_metadata": {"action_contract": "map-navigation-prior-v1"}}
    validate_guide_action_contract(payload, "map-navigation-prior-v1")
    with pytest.raises(ValueError, match="Guide action contract mismatch"):
        validate_guide_action_contract(payload, "current")


def test_guide_reward_contract_is_loaded_exactly_from_checkpoint() -> None:
    expected = RewardConfig(player_damage=0, death=0)
    payload = {"checkpoint_metadata": {"reward": expected.specification()}}
    assert guide_reward_config(payload) == expected
    with pytest.raises(ValueError, match="no reward contract"):
        guide_reward_config({"checkpoint_metadata": {}})


def test_natural_prefix_identity_binds_exact_guide_bytes(tmp_path: Path) -> None:
    guide = tmp_path / "guide.pt"
    guide.write_bytes(b"guide-v1")
    identity = natural_prefix_identity(NaturalPrefixConfig(target_phase=3), guide)

    assert identity["schema_version"] == 2
    assert identity["kind"] == "death-metal-natural-prefix-v2"
    assert identity["target_phase"] == 3
    assert identity["guide_checkpoint"] == str(guide.resolve())
    assert identity["guide_checkpoint_sha256"] == hashlib.sha256(b"guide-v1").hexdigest()


def test_natural_prefix_policy_sample_depends_on_seed_attempt_and_turn_only() -> None:
    sample = natural_prefix_policy_sample(17, 81001, 2, 9)
    assert sample == natural_prefix_policy_sample(17, 81001, 2, 9)
    assert sample != natural_prefix_policy_sample(17, 81002, 2, 9)
    assert sample != natural_prefix_policy_sample(17, 81001, 3, 9)
    assert sample != natural_prefix_policy_sample(17, 81001, 2, 10)
    with pytest.raises(ValueError, match="non-negative"):
        natural_prefix_policy_sample(17, 81001, -1, 9)
