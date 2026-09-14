import numpy as np
import pytest
import torch

from autodancer.constants import GRID_SIZE, Action, GridChannel, Terrain
from autodancer.live.protocol import ProtocolError, decode_observation
from autodancer.outcomes import classify_action_outcome
from autodancer.training.action_contract import (
    ActionContractMemory,
    UnsupportedLoadoutError,
    apply_action_contract,
)
from tests.test_dagger_controls import state


def bounded(armed=False):
    obs = state(armed)
    obs["bounded_loadout"] = np.array([1, 1], dtype=np.int8)
    return obs


def test_bounded_contract_preserves_directions_wait_and_bomb():
    obs = bounded()
    obs["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = Terrain.WALL
    memory = ActionContractMemory("bounded-bard-v1", 1)
    memory.observe(0, obs, Action.RIGHT, obs, {"action_outcome": {"category": "wall_attempt"}})
    assert np.all(memory.apply_slot(0, obs)["action_mask"][:6] == 1)
    armed = bounded(True)
    assert memory.apply_slot(0, armed)["action_mask"][Action.THROW] == 0


@pytest.mark.parametrize("contract", ["bounded-bard-v1", "bounded-bard-navigation-v1"])
def test_scope_violation_rejects_reset_batch_and_terminal_transition(contract):
    obs = bounded()
    bad = bounded()
    bad["bounded_loadout"][1] = 0
    memory = ActionContractMemory(contract, 1)
    with pytest.raises(UnsupportedLoadoutError):
        memory.reset_slot(0, bad)
    with pytest.raises(UnsupportedLoadoutError):
        memory.observe(0, obs, Action.RIGHT, bad, {"episode_status": "curriculum_complete"})
    batch = {k: np.stack([obs[k], bad[k]]) for k in obs}
    with pytest.raises(UnsupportedLoadoutError):
        ActionContractMemory(contract, 2).reset_batch(batch)


def test_bounded_navigation_reuses_dagger_navigation_and_clears_slot_memory():
    from autodancer.constants import MAP_SIZE, MapChannel

    obs = bounded()
    centre = MAP_SIZE // 2
    obs["map_memory"][centre, centre + 1, MapChannel.VISIT_COUNT] = 9
    wrapped = ActionContractMemory("bounded-bard-navigation-v1", 2)
    reference = ActionContractMemory("dagger-controls-v1", 2)
    batch = {key: np.stack([value, value]) for key, value in obs.items()}
    for key, value in reference.reset_batch(batch).items():
        np.testing.assert_array_equal(wrapped.reset_batch(batch)[key], value)
    assert wrapped.apply_slot(0, obs)["action_mask"][Action.RIGHT] == 0
    assert (
        ActionContractMemory("bounded-bard-v1", 1).apply_slot(0, obs)["action_mask"][Action.RIGHT]
        == 1
    )

    obs["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = Terrain.WALL
    info = {"action_outcome": {"category": "wall_attempt"}}
    actual = wrapped.observe(0, obs, Action.RIGHT, obs, info)
    expected = reference.observe(0, obs, Action.RIGHT, obs, info)
    assert actual == {**expected, "name": "bounded-bard-navigation-v1"}
    assert wrapped.masked_directions(0, obs) == (int(Action.RIGHT),)
    assert wrapped.masked_directions(1, obs) == ()
    assert wrapped.reset_slot(0, obs)["action_mask"][Action.RIGHT] == 1
    assert wrapped.masked_directions(0, obs) == ()


def test_bounded_navigation_does_not_mask_armed_throw_directions():
    obs = bounded(True)
    memory = ActionContractMemory("bounded-bard-navigation-v1", 1)
    result = memory.reset_slot(0, obs)
    assert np.all(result["action_mask"][:5] == 1)
    assert result["action_mask"][Action.THROW] == 0
    assert result["inventory"][0, 7] == 1
    diagnostic = memory.observe(
        0, obs, Action.RIGHT, obs, {"action_outcome": {"category": "wall_attempt"}}
    )
    assert not diagnostic["newly_learned_invalid_wall"]
    assert not diagnostic["navigation_prior_active"]


def test_missing_scope_does_not_silently_claim_support():
    obs = decode_observation(state())
    assert obs["bounded_loadout"].tolist() == [0, 0]
    with pytest.raises(ValueError, match="requires"):
        apply_action_contract(obs, "bounded-bard-v1")
    assert apply_action_contract(obs, "current") is obs


@pytest.mark.parametrize("value", [[0, 1], [2, 1], [1], [-1, 0]])
def test_invalid_scope_is_rejected(value):
    obs = bounded()
    obs["bounded_loadout"] = value
    with pytest.raises(ProtocolError):
        decode_observation(obs)


def test_bomb_placement_requires_inventory_consumption_and_visible_bomb():
    before, after = bounded(), bounded()
    before["inventory"][6] = [3, 42, 1, 0, 0, 0, 1, 0]
    after["grid"][GRID_SIZE // 2, GRID_SIZE // 2, GridChannel.STATUS] = 1
    outcome = classify_action_outcome(before, after, Action.BOMB, {})
    assert outcome.category == "bomb_placed" and not outcome.productive
    after["grid"][..., GridChannel.STATUS] = 0
    assert classify_action_outcome(before, after, Action.BOMB, {}).equipment_action is None


def test_collector_propagates_scope_violation_without_scoring_or_recovering():
    from autodancer.training.async_collector import VersionedAsyncRolloutCollector
    from autodancer.training.model import ModelConfig, RecurrentActorCritic
    from tests.test_async_collector import AsyncEnvironment

    environment = AsyncEnvironment()
    for worker in environment.environments.values():
        worker.reset = lambda **kwargs: (bounded(), {"seed": kwargs["seed"]})
        bad = bounded()
        bad["bounded_loadout"][1] = 0
        worker.step = lambda action, bad=bad: (
            bad,
            1.0,
            True,
            False,
            {"episode_status": "curriculum_complete"},
        )
    model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
            map_size=0,
        )
    )
    collector = VersionedAsyncRolloutCollector(
        environment,
        model,
        device=torch.device("cpu"),
        seed=81,
        action_contract="bounded-bard-v1",
    )
    try:
        with pytest.raises(UnsupportedLoadoutError):
            collector.collect(1)
        assert not collector.completed_episodes
        assert not environment.infrastructure_events
    finally:
        collector.close()
