"""Policy-side action-contract variants used by controlled experiments."""

from __future__ import annotations

import numpy as np

from autodancer.constants import Action

ACTION_CONTRACTS = ("current", "legacy-no-wait")


def apply_action_contract[Array: np.ndarray](
    observation: dict[str, Array], contract: str
) -> dict[str, Array]:
    """Return an effective policy observation without mutating live telemetry."""
    if contract not in ACTION_CONTRACTS:
        raise ValueError(f"Unknown action contract: {contract}")
    if contract == "current":
        return observation
    result = dict(observation)
    mask = observation["action_mask"].copy()
    mask[..., int(Action.WAIT)] = 0
    result["action_mask"] = mask
    return result
