from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import autodancer  # noqa: F401
from autodancer.constants import Action, ActorKind, GridChannel, Terrain, TrapKind
from autodancer.envs.sim import AutoDancerSimEnv
from autodancer.generator import ENEMY_REGISTRY
from autodancer.model import Actor
from autodancer.tasks import TASKS


def test_registered_sim_environment_passes_gymnasium_check() -> None:
    environment = gym.make("AutoDancer-Sim-v0", task="navigation").unwrapped
    check_env(environment, skip_render_check=True)


@pytest.mark.parametrize("task", TASKS)
def test_all_tasks_use_shared_schema(task: str) -> None:
    environment = AutoDancerSimEnv(task=task, render_mode="rgb_array")
    observation, info = environment.reset(seed=17)
    assert environment.observation_space.contains(observation)
    assert observation["grid"].shape == (21, 21, 7)
    assert observation["action_mask"].shape == (11,)
    assert info["task"] == task
    assert environment.render().shape == (256, 256, 3)


def test_same_seed_and_actions_replay_exactly() -> None:
    left = AutoDancerSimEnv(task="navigation")
    right = AutoDancerSimEnv(task="navigation")
    left_observation, _ = left.reset(seed=1234)
    right_observation, _ = right.reset(seed=1234)
    for key in left_observation:
        np.testing.assert_array_equal(left_observation[key], right_observation[key])
    assert left.state_digest() == right.state_digest()

    actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP, Action.WAIT] * 4
    for action in actions:
        left_result = left.step(action)
        right_result = right.step(action)
        for key in left_result[0]:
            np.testing.assert_array_equal(left_result[0][key], right_result[0][key])
        assert left_result[1:] == right_result[1:]


def test_observation_does_not_expose_dynamic_hidden_data() -> None:
    environment = AutoDancerSimEnv(task="all_zones")
    observation, _ = environment.reset(seed=9)
    hidden = observation["grid"][..., GridChannel.VISIBILITY] == 0
    assert np.all(observation["grid"][..., GridChannel.ACTOR][hidden] == 0)
    assert np.all(observation["grid"][..., GridChannel.HEALTH][hidden] == 0)
    assert np.all(observation["grid"][..., GridChannel.ITEM][hidden] == 0)
    assert np.all(observation["grid"][..., GridChannel.TRAP][hidden] == 0)


def test_melee_combat_emits_raw_events_and_rewards_separately() -> None:
    environment = AutoDancerSimEnv(task="single_enemy")
    environment.reset(seed=3)
    state = environment.state
    assert state is not None
    enemy = next(iter(state.enemies.values()))
    enemy.x, enemy.y = state.player.x + 1, state.player.y
    enemy.health = enemy.max_health = 1

    _, reward, terminated, truncated, info = environment.step(Action.RIGHT)
    event_kinds = [event["kind"] for event in info["raw_events"]]
    assert event_kinds == ["enemy_damage", "enemy_kill", "success"]
    assert reward == pytest.approx(-0.001 + 0.05 + 0.20 + 1.0)
    assert terminated and not truncated


def test_all_zones_progresses_through_regular_and_boss_floors() -> None:
    environment = AutoDancerSimEnv(task="all_zones")
    environment.reset(seed=44)
    visited: list[tuple[int, int]] = []
    for _ in range(16):
        state = environment.state
        assert state is not None
        visited.append((state.zone, state.floor))
        state.enemies.clear()
        state.player.x, state.player.y = state.stairs
        _, _, terminated, truncated, _ = environment.step(Action.WAIT)
        assert not truncated
    assert visited == [(zone, floor) for zone in range(1, 5) for floor in range(1, 5)]
    assert terminated
    assert environment.state is not None and environment.state.won


def test_action_mask_tracks_bombs_and_inventory() -> None:
    environment = AutoDancerSimEnv(task="navigation")
    observation, _ = environment.reset(seed=2)
    assert observation["action_mask"][Action.BOMB] == 1
    observation, *_ = environment.step(Action.BOMB)
    assert observation["action_mask"][Action.BOMB] == 0


def test_digging_consumes_a_turn_before_entering_the_tile() -> None:
    """Matches the Bard live trace captured on game build 22938426."""
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=2)
    state = environment.state
    assert state is not None
    state.enemies.clear()
    state.player.x, state.player.y = 2, 2
    state.terrain[1, 2] = Terrain.WALL

    _, _, _, _, info = environment.step(Action.UP)
    assert state.player.position == (2, 2)
    assert state.terrain[1, 2] == Terrain.FLOOR
    assert [event["kind"] for event in info["raw_events"]][:1] == ["wall_dug"]

    environment.step(Action.UP)
    assert state.player.position == (2, 1)


def test_zone_one_enemy_stats_match_live_prototypes() -> None:
    definitions = {definition.kind: definition for definition in ENEMY_REGISTRY[1]}
    assert definitions[ActorKind.GREEN_SLIME].health == 1
    assert definitions[ActorKind.BLUE_SLIME].health == 2
    assert definitions[ActorKind.BLUE_SLIME].damage == 2
    assert definitions[ActorKind.ZOMBIE].health == 1
    assert definitions[ActorKind.ZOMBIE].damage == 2
    assert definitions[ActorKind.SKELETON].health == 1


def test_live_slime_patterns_and_first_turn_cadence() -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=2)
    state = environment.state
    assert state is not None
    state.player.x, state.player.y = 6, 5
    green = Actor(2, ActorKind.GREEN_SLIME, 3, 5, 1, 1, move_period=1)
    blue = Actor(3, ActorKind.BLUE_SLIME, 5, 5, 2, 2, move_period=2)
    state.enemies = {green.entity_id: green, blue.entity_id: blue}

    environment.step(Action.WAIT)
    assert green.position == (3, 5)
    assert blue.position == (5, 6)
    assert state.player.health == state.player.max_health

    environment.step(Action.WAIT)
    assert blue.position == (5, 6)

    environment.step(Action.WAIT)
    assert blue.position == (5, 5)


def test_zombie_moves_linearly_on_first_and_third_turns() -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=2)
    state = environment.state
    assert state is not None
    state.player.x, state.player.y = 10, 10
    zombie = Actor(
        2, ActorKind.ZOMBIE, 4, 4, 1, 1, move_period=2, facing=1
    )
    state.enemies = {zombie.entity_id: zombie}

    environment.step(Action.WAIT)
    assert zombie.position == (5, 4)
    environment.step(Action.WAIT)
    assert zombie.position == (5, 4)
    environment.step(Action.WAIT)
    assert zombie.position == (6, 4)


def test_bounce_trap_forces_player_in_its_direction() -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=2)
    state = environment.state
    assert state is not None
    state.enemies.clear()
    state.player.x, state.player.y = 4, 4
    state.traps[4, 3] = TrapKind.BOUNCE_RIGHT

    _, _, _, _, info = environment.step(Action.LEFT)
    assert state.player.position == (4, 4)
    assert "trap_activated" in [event["kind"] for event in info["raw_events"]]

    environment.step(Action.LEFT)
    assert state.player.position == (3, 4)
    environment.step(Action.RIGHT)
    environment.step(Action.LEFT)
    assert state.player.position == (4, 4)


def test_enemy_gold_drops_are_collected_from_the_ground() -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=2)
    state = environment.state
    assert state is not None
    state.player.x, state.player.y = 4, 4
    enemy = Actor(2, ActorKind.BLUE_SLIME, 5, 4, 1, 2, move_period=2)
    state.enemies = {enemy.entity_id: enemy}

    environment.step(Action.RIGHT)
    assert state.gold == 0
    assert state.items[(5, 4)].kind == 1
    assert state.items[(5, 4)].value == 4

    _, _, _, _, info = environment.step(Action.RIGHT)
    assert state.gold == 4
    assert "item_collected" in [event["kind"] for event in info["raw_events"]]


def test_bomb_explodes_three_turns_after_placement() -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=2)
    state = environment.state
    assert state is not None
    state.enemies.clear()
    state.player.x, state.player.y = 4, 4

    environment.step(Action.BOMB)
    assert len(state.active_bombs) == 1
    environment.step(Action.RIGHT)
    assert len(state.active_bombs) == 1
    environment.step(Action.RIGHT)
    assert len(state.active_bombs) == 1
    environment.step(Action.RIGHT)
    assert state.active_bombs == []


def test_broadsword_hits_three_tiles_in_front_without_moving() -> None:
    environment = AutoDancerSimEnv(task="navigation")
    environment.reset(seed=2)
    state = environment.state
    assert state is not None
    state.player.x, state.player.y = 5, 5
    state.inventory[0] = (8, 1, 1)
    state.enemies = {
        index: Actor(index, ActorKind.GREEN_SLIME, x, 4, 1, 1)
        for index, x in enumerate((4, 5, 6), start=2)
    }

    environment.step(Action.UP)
    assert state.player.position == (5, 5)
    assert state.enemies == {}
