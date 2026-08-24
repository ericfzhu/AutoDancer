from __future__ import annotations

import numpy as np

from autodancer.constants import ACTION_COUNT, Action
from autodancer.training.action_contract import apply_action_contract


def test_legacy_contract_masks_wait_without_mutating_live_observation() -> None:
    live = {"action_mask": np.ones((2, ACTION_COUNT), dtype=np.int8)}
    legacy = apply_action_contract(live, "legacy-no-wait")
    assert np.all(live["action_mask"][:, int(Action.WAIT)] == 1)
    assert np.all(legacy["action_mask"][:, int(Action.WAIT)] == 0)
    assert np.all(legacy["action_mask"][:, :4] == 1)


def test_current_contract_returns_live_observation_unchanged() -> None:
    live = {"action_mask": np.ones(ACTION_COUNT, dtype=np.int8)}
    assert apply_action_contract(live, "current") is live
