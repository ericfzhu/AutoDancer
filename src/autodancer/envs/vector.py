"""Ordered fixed-capacity vector adapter for live game workers."""

from __future__ import annotations

import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from autodancer.live.supervisor import AutoDancerSupervisor
from autodancer.memory import MapCapacityError


class AutoDancerVectorEnv:
    def __init__(self, supervisor: AutoDancerSupervisor) -> None:
        self.supervisor = supervisor
        self.worker_ids = supervisor.worker_ids
        self.environments = {
            worker_id: supervisor.environment(worker_id) for worker_id in self.worker_ids
        }
        self._executor = ThreadPoolExecutor(max_workers=len(self.worker_ids))

    @property
    def num_envs(self) -> int:
        return len(self.worker_ids)

    @staticmethod
    def _stack(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            key: np.stack([observation[key] for observation in observations])
            for key in observations[0]
        }

    def reset(
        self, seeds: list[int] | np.ndarray
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} seeds, received {len(seeds)}")
        futures = [
            self._executor.submit(self.environments[worker_id].reset, seed=int(seed))
            for worker_id, seed in zip(self.worker_ids, seeds, strict=True)
        ]
        results = []
        for index, future in enumerate(futures):
            try:
                results.append(future.result())
            except MapCapacityError:
                raise
            except Exception:
                results.append(self.recover(index, secrets.randbelow(2**31)))
        return self._stack([result[0] for result in results]), [result[1] for result in results]

    def step(
        self, actions: list[int] | np.ndarray
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, received {len(actions)}")
        started = time.monotonic()
        futures = [
            self._executor.submit(self.environments[worker_id].step, int(action))
            for worker_id, action in zip(self.worker_ids, actions, strict=True)
        ]
        results = []
        for index, future in enumerate(futures):
            try:
                results.append(future.result())
            except MapCapacityError:
                raise
            except Exception as error:
                observation, reset_info = self.recover(index, secrets.randbelow(2**31))
                results.append(
                    (
                        observation,
                        -1.0,
                        False,
                        True,
                        {
                            **reset_info,
                            "episode_status": "aborted",
                            "worker_replaced": True,
                            "failure": type(error).__name__,
                            "bridge": None,
                            "reward_components": {"worker_failure": -1.0},
                        },
                    )
                )
        latency = time.monotonic() - started
        for worker_id, result in zip(self.worker_ids, results, strict=True):
            handle = self.supervisor.workers[worker_id]
            handle.last_latency = latency
            handle.episode_status = str(result[4].get("episode_status", "unknown"))
            acknowledgement = result[4].get("bridge") or {}
            handle.last_acknowledged_command = int(acknowledgement.get("command_id", 0))
        return (
            self._stack([result[0] for result in results]),
            np.asarray([result[1] for result in results], dtype=np.float32),
            np.asarray([result[2] for result in results], dtype=np.bool_),
            np.asarray([result[3] for result in results], dtype=np.bool_),
            [result[4] for result in results],
        )

    def reset_at(
        self, indices: list[int], seeds: list[int]
    ) -> list[tuple[dict[str, np.ndarray], dict[str, Any]]]:
        if len(indices) != len(seeds):
            raise ValueError("indices and seeds must have the same length")
        futures = [
            self._executor.submit(
                self.environments[self.worker_ids[index]].reset,
                seed=int(seed),
            )
            for index, seed in zip(indices, seeds, strict=True)
        ]
        results = []
        for index, seed, future in zip(indices, seeds, futures, strict=True):
            try:
                results.append(future.result())
            except MapCapacityError:
                raise
            except Exception:
                results.append(self.recover(index, int(seed)))
        return results

    def recover(self, worker_index: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        worker_id = self.worker_ids[worker_index]
        self.supervisor.replace_worker(worker_id)
        self.environments[worker_id].close()
        environment = self.supervisor.environment(worker_id)
        self.environments[worker_id] = environment
        return environment.reset(seed=seed)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        for environment in self.environments.values():
            environment.close()
        self.supervisor.close()
