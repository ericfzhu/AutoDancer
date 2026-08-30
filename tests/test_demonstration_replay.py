from __future__ import annotations

from typing import Any

import numpy as np

from autodancer.constants import ACTION_COUNT, PLAYER_FEATURES, PlayerFeature
from autodancer.training.demonstration_replay import replay_trace


class FakeEnvironment:
    def __init__(self) -> None:
        self.turn = 0

    @staticmethod
    def observation(zone: int, floor: int) -> dict[str, np.ndarray]:
        player = np.zeros(PLAYER_FEATURES, dtype=np.int32)
        player[PlayerFeature.ZONE] = zone
        player[PlayerFeature.FLOOR] = floor
        return {
            "player": player,
            "action_mask": np.ones(ACTION_COUNT, dtype=np.int8),
        }

    def reset(
        self, *, seed: int, options: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        assert seed == 92043
        assert options["curriculum"]["start_level"] == 4
        self.turn = 0
        return self.observation(1, 4), {"zone": 1, "floor": 4}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert action == [0, 1, 5][self.turn]
        self.turn += 1
        terminal = self.turn == 3
        zone, floor = (2, 1) if terminal else (1, 4)
        events = [{"kind": "enemy_damage"}]
        if terminal:
            events.append({"kind": "enemy_kill"})
        return (
            self.observation(zone, floor),
            0.0,
            terminal,
            False,
            {
                "zone": zone,
                "floor": floor,
                "episode_status": "curriculum_complete" if terminal else "running",
                "raw_events": events,
            },
        )


def trace() -> dict[str, Any]:
    return {
        "trace_id": "a" * 64,
        "seed": 92043,
        "status": "curriculum_complete",
        "turns": 3,
        "furthest_zone": 2,
        "furthest_floor": 1,
        "curriculum_reset": {
            "id": "full-boss",
            "start_level": 4,
            "target_level": 5,
            "profile": "player20",
        },
        "event_counts": {"enemy_damage": 3, "enemy_kill": 1},
        "action_sequence": [0, 1, 5],
    }


def test_replay_trace_requires_matching_terminal_gameplay_evidence() -> None:
    result = replay_trace(FakeEnvironment(), trace())
    assert result["valid"] is True
    assert all(result["checks"].values())


def test_replay_trace_rejects_terminal_summary_mismatch() -> None:
    candidate = trace()
    candidate["event_counts"] = {"enemy_damage": 2, "enemy_kill": 1}
    result = replay_trace(FakeEnvironment(), candidate)
    assert result["valid"] is False
    assert result["checks"]["event_counts"] is False


def test_replay_trace_rejects_action_that_is_now_masked() -> None:
    environment = FakeEnvironment()
    original = environment.observation

    def masked(zone: int, floor: int) -> dict[str, np.ndarray]:
        value = original(zone, floor)
        value["action_mask"][0] = 0
        return value

    environment.observation = masked  # type: ignore[method-assign]
    result = replay_trace(environment, trace())
    assert result["valid"] is False
    assert "masked" in result["error"]
