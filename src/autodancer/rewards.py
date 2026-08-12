"""Reward calculation that is shared by simulator and live records."""

from collections.abc import Iterable, Mapping
from typing import Any

from autodancer.tasks import REWARD_VALUES


def reward_from_event_dicts(
    events: Iterable[Mapping[str, Any]], values: Mapping[str, float] | None = None
) -> float:
    configured = REWARD_VALUES | dict(values or {})
    reward = configured["turn"]
    for event in events:
        kind = str(event.get("kind", ""))
        amount = int(event.get("amount", 0))
        if kind == "success":
            reward += configured["success"]
        elif kind == "failure":
            reward += configured["failure"]
        elif kind == "enemy_damage":
            reward += configured["enemy_damage"] * amount
        elif kind == "enemy_kill":
            reward += configured["enemy_kill"]
        elif kind == "player_damage":
            reward += configured["player_damage"] * amount
        elif kind == "reveal":
            reward += configured["reveal"] * amount
    return float(reward)

