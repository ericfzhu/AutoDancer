from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from autodancer.constants import BossType, PlayerFeature
from autodancer.curriculum import EpisodeResetSpec
from autodancer.training.boss_identity import collect_boss_identities


class ResetOnlyFakeEnvironment:
    num_envs = 3
    worker_ids = ["worker-0000", "worker-0001", "worker-0002"]
    infrastructure_events: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.reset_batches: list[list[int]] = []
        self.step_calls = 0

    def reset(self, seeds: list[int], options=None):
        self.reset_batches.append(list(seeds))
        players = np.zeros((self.num_envs, 21), dtype=np.int64)
        infos = []
        for slot, seed in enumerate(seeds):
            boss_type = int(BossType.DEATH_METAL if seed % 2 else BossType.KING_CONGA)
            players[slot, PlayerFeature.TASK] = boss_type
            infos.append(
                {
                    "seed": seed,
                    "instance_id": self.worker_ids[slot],
                    "character": "Bard",
                    "episode_status": "running",
                    "boss_type": boss_type,
                    "run_id": f"run-{seed}",
                    "session_id": "session",
                    "launch_id": f"launch-{slot}",
                    "curriculum_reset": options[slot]["curriculum"],
                }
            )
        return {"player": players}, infos

    def step(self, actions):
        self.step_calls += 1
        raise AssertionError(f"reset-only calibration attempted actions {actions}")


def test_boss_identity_calibration_never_steps_and_preserves_seed_order() -> None:
    environment = ResetOnlyFakeEnvironment()
    reset_spec = EpisodeResetSpec("boss-identity", 4, 5, "player20")

    results = collect_boss_identities(environment, [11, 12, 13, 14], reset_spec)

    assert [result["seed"] for result in results] == [11, 12, 13, 14]
    assert [result["boss_name"] for result in results] == [
        "DEATH_METAL",
        "KING_CONGA",
        "DEATH_METAL",
        "KING_CONGA",
    ]
    assert environment.step_calls == 0
    assert len(environment.reset_batches) == 2
    assert environment.reset_batches[1][0] == 14
    assert len(environment.reset_batches[1]) == environment.num_envs


def test_boss_identity_calibration_rejects_cross_worker_record() -> None:
    environment = ResetOnlyFakeEnvironment()
    original_reset = environment.reset

    def crossed_reset(seeds: list[int], options=None):
        observation, infos = original_reset(seeds, options=options)
        infos[0]["instance_id"] = "worker-0001"
        return observation, infos

    environment.reset = crossed_reset  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="worker mismatch"):
        collect_boss_identities(
            environment,
            [11, 12, 13],
            EpisodeResetSpec("boss-identity", 4, 5, "player20"),
        )


def test_boss_identity_calibration_rejects_observation_info_disagreement() -> None:
    environment = ResetOnlyFakeEnvironment()
    original_reset = environment.reset

    def mismatched_reset(seeds: list[int], options=None):
        observation, infos = original_reset(seeds, options=options)
        infos[0]["boss_type"] = int(BossType.CORAL_RIFF)
        return observation, infos

    environment.reset = mismatched_reset  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="boss identity disagreement"):
        collect_boss_identities(
            environment,
            [11, 12, 13],
            EpisodeResetSpec("boss-identity", 4, 5, "player20"),
        )
