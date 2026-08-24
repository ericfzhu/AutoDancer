from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    PLAYER_FEATURES,
    Action,
    ActorKind,
    GridChannel,
    PlayerFeature,
    Terrain,
)
from autodancer.live.diagnose import ProbeLedger
from autodancer.live.protocol import ProtocolError, validate_record
from autodancer.outcomes import classify_action_outcome
from tests.test_live_protocol import record


def _observation(*, x: int = 0, y: int = 0) -> dict[str, np.ndarray]:
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16)
    grid[..., GridChannel.TERRAIN_CLASS] = Terrain.FLOOR
    player = np.zeros(PLAYER_FEATURES, dtype=np.int32)
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 1
    player[PlayerFeature.X] = x
    player[PlayerFeature.Y] = y
    return {
        "grid": grid,
        "player": player,
        "inventory": np.zeros((INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16),
        "action_mask": np.ones(ACTION_COUNT, dtype=np.int8),
    }


def test_lua_bridge_maps_all_logical_actions_in_python_order() -> None:
    source = Path("mods/AutoDancer/scripts/Bridge.lua").read_text(encoding="utf-8")
    expected = {
        Action.UP: "Action.Direction.UP",
        Action.RIGHT: "Action.Direction.RIGHT",
        Action.DOWN: "Action.Direction.DOWN",
        Action.LEFT: "Action.Direction.LEFT",
        Action.WAIT: "Action.Special.IDLE",
        Action.BOMB: "Action.Special.BOMB",
        Action.ITEM_1: "Action.Special.ITEM_1",
        Action.ITEM_2: "Action.Special.ITEM_2",
        Action.THROW: "Action.Special.THROW",
        Action.SPELL_1: "Action.Special.SPELL_1",
        Action.SPELL_2: "Action.Special.SPELL_2",
    }
    for action, engine_name in expected.items():
        assert f"[{int(action)}] = {engine_name}" in source


def test_running_mask_requires_wait_and_matches_inventory() -> None:
    payload = record(0, "reset")
    validate_record(payload)
    payload["observation"]["action_mask"][Action.WAIT] = 0
    with pytest.raises(ProtocolError, match="directions and WAIT"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["observation"]["inventory"][6][0] = 3
    with pytest.raises(ProtocolError, match="inventory availability"):
        validate_record(payload)
    payload["observation"]["action_mask"][Action.BOMB] = 1
    validate_record(payload)


def test_action_acknowledgement_requires_observed_engine_action() -> None:
    payload = record(1, "turn", requested_action=Action.WAIT)
    payload["bridge"]["observed_action"] = None
    with pytest.raises(ProtocolError, match="observed_action"):
        validate_record(payload)
    payload["bridge"]["observed_action"] = 2
    payload["bridge"]["engine_action"] = 1
    with pytest.raises(ProtocolError, match="does not match"):
        validate_record(payload)


def test_probe_ledger_preserves_first_successful_action_evidence() -> None:
    ledger = ProbeLedger()
    first = {
        "bridge": {"requested_action": 0, "engine_action": 3, "observed_action": 3},
        "action_outcome": {"category": "move"},
    }
    later = {
        "bridge": {"requested_action": 0, "engine_action": 3, "observed_action": 3},
        "action_outcome": {"category": "combat"},
    }
    ledger.observe_step(Action.UP, first, seed=46_000)
    ledger.observe_step(Action.UP, later, seed=46_001)
    evidence = ledger.actions[Action.UP]
    assert evidence.acknowledged
    assert evidence.seed == 46_000
    assert evidence.outcome == "move"


def test_outcomes_separate_move_wait_combat_wall_dig_and_floor_transition() -> None:
    before = _observation()
    moved = _observation(x=1)
    assert classify_action_outcome(before, moved, Action.RIGHT, {}).category == "move"
    assert classify_action_outcome(before, before, Action.WAIT, {}).category == "wait"
    combat_info = {"raw_events": [{"kind": "enemy_damage", "amount": 1}]}
    assert classify_action_outcome(before, before, Action.RIGHT, combat_info).category == "combat"

    wall = _observation()
    wall["grid"][10, 11, GridChannel.TERRAIN_CLASS] = Terrain.WALL
    assert classify_action_outcome(wall, wall, Action.RIGHT, {}).category == "wall_attempt"
    dug = _observation()
    assert classify_action_outcome(wall, dug, Action.RIGHT, {}).category == "dig"

    enemy = _observation()
    enemy["grid"][10, 11, GridChannel.ACTOR_CLASS] = ActorKind.GREEN_SLIME
    assert classify_action_outcome(enemy, enemy, Action.RIGHT, {}).category == "combat_attempt"

    next_floor = _observation()
    next_floor["player"][PlayerFeature.FLOOR] = 2
    result = classify_action_outcome(before, next_floor, Action.DOWN, {})
    assert result.category == "floor_transition"
    assert result.floor_changed
