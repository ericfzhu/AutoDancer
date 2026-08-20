from __future__ import annotations

from autodancer.constants import ActorKind, PlayerFeature
from autodancer.envs.sim import AutoDancerSimEnv
from autodancer.model import Actor
from autodancer.observation import encode_observation
from autodancer.rewards import reward_from_event_dicts


def test_observation_counts_only_visible_enemies() -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=4)
    state = environment.state
    assert state is not None
    state.enemies = {
        2: Actor(2, ActorKind.GREEN_SLIME, state.player.x + 1, state.player.y, 1, 1),
        3: Actor(3, ActorKind.GREEN_SLIME, state.player.x + 12, state.player.y, 1, 1),
    }
    state.explored.fill(False)
    state.visible.fill(False)
    state.explored[state.player.y, state.player.x] = True
    state.visible[state.player.y, state.player.x] = True
    state.explored[state.player.y, state.player.x + 1] = True
    state.visible[state.player.y, state.player.x + 1] = True
    observation = encode_observation(state)
    assert observation["player"][PlayerFeature.VISIBLE_ENEMIES] == 1


def test_dead_enemy_cannot_emit_negative_damage_or_reward() -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=2)
    enemy = Actor(2, ActorKind.GREEN_SLIME, 5, 5, 1, 1)
    first = environment._damage_enemy(enemy, 4, "melee")
    second = environment._damage_enemy(enemy, 4, "bomb")
    assert first[0].amount == 1
    assert enemy.health == 0
    assert second == []
    assert reward_from_event_dicts(event.to_dict() for event in first) == 0.049


def test_snapshot_and_digest_cover_future_relevant_state() -> None:
    left = AutoDancerSimEnv(task="floor")
    right = AutoDancerSimEnv(task="floor")
    left.reset(seed=9)
    right.reset(seed=9)
    assert left.state_digest() == right.state_digest()
    snapshot = left.snapshot()
    assert {"terrain", "traps", "inventory", "active_bombs", "rng"} <= snapshot.keys()
    left._require_channels().channel("enemy_ai").integers(0, 4)
    assert left.state_digest() != right.state_digest()
