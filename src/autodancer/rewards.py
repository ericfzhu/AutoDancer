"""Reward calculation for raw live-game events."""

from collections.abc import Iterable, Mapping
from typing import Any

REWARD_VALUES = {
    "turn": -0.001,
    "success": 1.0,
    "failure": -1.0,
    "enemy_damage": 0.05,
    "enemy_kill": 0.1,
    "player_damage": -0.1,
    "reveal": 0.001,
}


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

