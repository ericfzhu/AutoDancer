from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import autodancer  # noqa: F401
from autodancer.constants import Action, GridChannel
from autodancer.envs.sim import AutoDancerSimEnv
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
