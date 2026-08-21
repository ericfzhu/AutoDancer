from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.supervisor import SupervisorError


def observation(value: int) -> dict[str, np.ndarray]:
    return {
        "grid": np.full((1, 1, 1), value, dtype=np.int16),
        "player": np.full(1, value, dtype=np.int32),
        "inventory": np.full((1, 1), value, dtype=np.int16),
        "action_mask": np.ones(1, dtype=np.int8),
    }


class FakeEnvironment:
    def __init__(self, slot: int, *, fail_once: bool = False) -> None:
        self.slot = slot
        self.fail_once = fail_once

    def reset(self, *, seed: int):
        return observation(self.slot), {"seed": seed, "episode_status": "running"}

    def step(self, action: int):
        if self.fail_once:
            self.fail_once = False
            raise TimeoutError("hung worker")
        return (
            observation(self.slot),
            float(self.slot + action),
            False,
            False,
            {"episode_status": "running", "bridge": {"command_id": action + 1}},
        )

    def close(self) -> None:
        pass


class FakeSupervisor:
    worker_ids = ["worker-0000", "worker-0001"]

    def __init__(self) -> None:
        self.workers = {
            worker_id: SimpleNamespace(
                last_latency=0.0, episode_status="", last_acknowledged_command=0
            )
            for worker_id in self.worker_ids
        }
        self.environments = {
            "worker-0000": FakeEnvironment(0),
            "worker-0001": FakeEnvironment(1, fail_once=True),
        }
        self.replacements: list[str] = []
        self.closed = False

    def environment(self, worker_id: str) -> FakeEnvironment:
        return self.environments[worker_id]

    def replace_worker(self, worker_id: str) -> None:
        self.replacements.append(worker_id)
        self.environments[worker_id] = FakeEnvironment(int(worker_id[-1]))

    def close(self) -> None:
        self.closed = True


def test_vector_env_preserves_slot_order_and_replaces_failed_worker() -> None:
    supervisor = FakeSupervisor()
    environment = AutoDancerVectorEnv(supervisor)  # type: ignore[arg-type]
    reset_observation, infos = environment.reset([11, 22])
    assert reset_observation["player"][:, 0].tolist() == [0, 1]
    assert [info["seed"] for info in infos] == [11, 22]

    next_observation, rewards, _, truncated, infos = environment.step([3, 4])
    assert next_observation["player"][:, 0].tolist() == [0, 1]
    assert rewards.tolist() == [3.0, -1.0]
    assert truncated.tolist() == [False, True]
    assert infos[1]["worker_replaced"] is True
    assert supervisor.replacements == ["worker-0001"]
    environment.close()
    assert supervisor.closed


def test_vector_env_propagates_fixed_capacity_replacement_failure() -> None:
    supervisor = FakeSupervisor()

    def fail_replacement(worker_id: str) -> None:
        raise SupervisorError(f"could not restore {worker_id}")

    supervisor.replace_worker = fail_replacement  # type: ignore[method-assign]
    environment = AutoDancerVectorEnv(supervisor)  # type: ignore[arg-type]
    with pytest.raises(SupervisorError, match="could not restore worker-0001"):
        environment.step([0, 0])
    environment.close()
