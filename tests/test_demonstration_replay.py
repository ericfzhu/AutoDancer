from __future__ import annotations

from typing import Any

import numpy as np

from autodancer.constants import PlayerFeature
from autodancer.observation import observation_space
from autodancer.rewards import RewardConfig
from autodancer.training.demonstration_replay import RecurrentReplayCapture, replay_trace
from autodancer.training.model import START_ACTION


class FakeEnvironment:
    def __init__(self) -> None:
        self.turn = 0

    @staticmethod
    def observation(zone: int, floor: int) -> dict[str, np.ndarray]:
        value = {
            name: np.zeros(space.shape, dtype=space.dtype)
            for name, space in observation_space().spaces.items()
        }
        player = value["player"]
        player[PlayerFeature.ZONE] = zone
        player[PlayerFeature.FLOOR] = floor
        value["action_mask"].fill(1)
        return value

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


class EarlyTerminalEnvironment(FakeEnvironment):
    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert action == [0, 1][self.turn]
        self.turn += 1
        terminal = self.turn == 2
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
    assert result["actual"]["final_action"] == 5
    assert result["actual"]["final_pre_action_observation"]["zone"] == 1
    assert result["actual"]["final_observation"]["zone"] == 2
    assert all(result["checks"].values())


def test_replay_trace_rejects_terminal_summary_mismatch() -> None:
    candidate = trace()
    candidate["event_counts"] = {"enemy_damage": 2, "enemy_kill": 1}
    result = replay_trace(FakeEnvironment(), candidate)
    assert result["valid"] is False
    assert result["checks"]["event_counts"] is False


def test_replay_trace_canonicalizes_only_a_proven_successful_prefix() -> None:
    candidate = trace()
    candidate["event_counts"] = {"enemy_damage": 2, "enemy_kill": 1}
    capture = RecurrentReplayCapture("current", RewardConfig(profile_version=5))

    result = replay_trace(
        EarlyTerminalEnvironment(), candidate, recurrent_capture=capture
    )

    assert result["valid"] is True
    assert result["checks"]["successful_action_prefix"] is True
    assert result["actual"]["qualified_action_sequence"] == [0, 1]
    assert result["actual"]["stale_suffix_action_count"] == 1
    assert len(result["turn_digests"]) == 3
    assert len(result["turn_component_digests"]) == 3
    assert set(result["turn_component_digests"][0]) == set(observation_space().spaces)
    assert capture.demonstration is not None
    assert capture.demonstration.actions.tolist() == [0, 1]


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


def test_replay_trace_captures_complete_recurrent_policy_inputs() -> None:
    capture = RecurrentReplayCapture("current", RewardConfig(profile_version=5))

    result = replay_trace(FakeEnvironment(), trace(), recurrent_capture=capture)

    assert result["valid"] is True
    assert capture.demonstration is not None
    demonstration = capture.demonstration
    assert demonstration.trace_id == "a" * 64
    assert demonstration.seed == 92043
    assert demonstration.length == 3
    assert demonstration.observations["grid"].shape[0] == 3
    assert demonstration.actions.tolist() == [0, 1, 5]
    assert demonstration.previous_actions.tolist() == [START_ACTION, 0, 1]
    assert demonstration.episode_starts.tolist() == [True, False, False]
    assert np.all(np.isfinite(demonstration.previous_rewards))
