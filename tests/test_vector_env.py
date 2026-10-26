from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autodancer.envs.vector import AutoDancerVectorEnv, VectorInfrastructureError
from autodancer.live.protocol import ProtocolError


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

    def reset(self, *, seed: int, options=None):
        return observation(self.slot), {
            "seed": seed,
            "episode_status": "running",
            "options": options,
        }

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

    def record_failure(self, worker_id, failure):
        self.failure = (worker_id, failure)

    def replace_worker(self, worker_id: str, *, failure=None) -> None:
        del failure
        self.replacements.append(worker_id)
        self.environments[worker_id] = FakeEnvironment(int(worker_id[-1]))

    def close(self) -> None:
        self.closed = True


def test_vector_env_preserves_slot_order_and_stops_on_failed_worker() -> None:
    supervisor = FakeSupervisor()
    environment = AutoDancerVectorEnv(supervisor)  # type: ignore[arg-type]
    reset_options = [
        {"curriculum": {"id": "left"}},
        {"curriculum": {"id": "right"}},
    ]
    reset_observation, infos = environment.reset([11, 22], options=reset_options)
    assert reset_observation["player"][:, 0].tolist() == [0, 1]
    assert [info["seed"] for info in infos] == [11, 22]
    assert [info["options"] for info in infos] == reset_options

    with pytest.raises(VectorInfrastructureError, match="worker-0001"):
        environment.step([3, 4])
    assert supervisor.replacements == []
    assert supervisor.failure[0] == "worker-0001"
    assert environment.infrastructure_events[0]["operation"] == "step"
    environment.close()
    assert supervisor.closed


@pytest.mark.parametrize("operation", ["reset", "reset_at"])
def test_vector_reset_failure_is_not_retried(operation):
    supervisor = FakeSupervisor()

    def invalid_reset(**kwargs):
        raise ProtocolError("Unknown world identity")

    supervisor.environments["worker-0001"].reset = invalid_reset
    environment = AutoDancerVectorEnv(supervisor)
    try:
        with pytest.raises(VectorInfrastructureError, match="Unknown world identity"):
            if operation == "reset":
                environment.reset([11, 22])
            else:
                environment.reset_at([1], [22])
        assert not supervisor.replacements
        assert supervisor.failure[1]["operation"] == operation
    finally:
        environment.close()
