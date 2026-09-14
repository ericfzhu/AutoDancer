from __future__ import annotations

import numpy as np
import pytest

from autodancer.constants import GRID_SIZE, Action, GridChannel, Terrain
from autodancer.live.protocol import ProtocolError, decode_observation
from autodancer.outcomes import classify_action_outcome
from autodancer.training.action_contract import ActionContractMemory, apply_action_contract
from tests.test_action_contract_modes import _observation


def state(armed=False):
    obs = _observation()
    obs["equipment_controls"] = np.array([1, 1, armed], dtype=np.int8)
    obs["inventory"][0, :2] = [5, 123]
    return obs


def test_projection_is_opt_in_and_batch_preserves_armed_inventory():
    before, armed = state(), state(True)
    assert apply_action_contract(armed, "map-navigation-prior-v1") is armed
    batch = {k: np.stack([before[k], armed[k]]) for k in before}
    effective = ActionContractMemory("dagger-controls-v1", 2).reset_batch(batch)
    assert effective["inventory"][:, 0, 7].tolist() == [0, 1]
    assert effective["action_mask"][:, Action.THROW].tolist() == [1, 0]
    assert np.all(batch["inventory"][:, 0, 7] == 0)
    assert np.all(batch["action_mask"][:, Action.THROW] == 1)


def test_armed_throw_bypasses_wall_memory_and_does_not_learn_walls():
    memory = ActionContractMemory("dagger-controls-v1", 1)
    before = state()
    before["grid"][GRID_SIZE // 2, GRID_SIZE // 2 + 1, GridChannel.TERRAIN_CLASS] = Terrain.WALL
    armed = {k: v.copy() for k, v in before.items()}
    armed["equipment_controls"][2] = 1
    info = {"action_outcome": {"category": "wall_attempt"}}
    memory.observe(0, before, Action.RIGHT, before, info)
    assert memory.apply_slot(0, before)["action_mask"][Action.RIGHT] == 0
    assert np.all(memory.apply_slot(0, armed)["action_mask"][:5] == 1)
    diagnostic = memory.observe(0, armed, Action.RIGHT, armed, info)
    assert not diagnostic["newly_learned_invalid_wall"]
    assert diagnostic["effective_masked_direction_count"] == 0
    assert not diagnostic["navigation_prior_active"]
    memory.reset_slot(0, before)
    assert memory.apply_slot(0, before)["action_mask"][Action.RIGHT] == 1


def test_unsupported_equipment_keeps_throw_and_toggle_state():
    obs = state()
    obs["equipment_controls"][:] = [1, 0, 0]
    obs["inventory"][0, 7] = 1
    effective = apply_action_contract(obs, "dagger-controls-v1")
    assert effective["inventory"][0, 7] == 1
    assert effective["action_mask"][Action.THROW] == 1


def test_missing_extension_fails_closed_for_new_contract_only():
    obs = _observation()
    decoded = decode_observation(obs)
    assert decoded["equipment_controls"].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="requires"):
        apply_action_contract(decoded, "dagger-controls-v1")
    assert apply_action_contract(decoded, "current") is decoded


def test_shared_throw_button_remains_available_with_additional_equipment():
    obs = state(True)
    obs["inventory"][10, 0] = 7  # Boots may use THROW as a toggle.
    effective = apply_action_contract(obs, "dagger-controls-v1")
    assert effective["action_mask"][Action.THROW] == 1
    assert effective["inventory"][0, 7] == 1


@pytest.mark.parametrize("controls", [[2, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1]])
def test_malformed_control_extension_is_rejected(controls):
    obs = state()
    obs["equipment_controls"] = controls
    with pytest.raises(ProtocolError):
        decode_observation(obs)


def test_equipment_outcomes_preserve_combat_and_legacy_behavior():
    before, armed = state(), state(True)
    thrown = state()
    thrown["equipment_controls"][:] = [1, 0, 0]
    thrown["inventory"][0] = 0
    assert classify_action_outcome(before, armed, Action.THROW, {}).category == "throw_prepared"
    assert classify_action_outcome(armed, armed, Action.THROW, {}).category == "throw_already_armed"
    assert classify_action_outcome(armed, thrown, Action.RIGHT, {}).category == "weapon_thrown"
    combat = classify_action_outcome(
        armed, thrown, Action.RIGHT, {"raw_events": [{"kind": "enemy_damage"}]}
    )
    assert combat.category == "combat" and combat.equipment_action == "weapon_thrown"
    assert classify_action_outcome(armed, armed, Action.WAIT, {}).category == "wait"
    before.pop("equipment_controls")
    armed.pop("equipment_controls")
    assert classify_action_outcome(before, armed, Action.THROW, {}).category == "special_no_effect"
