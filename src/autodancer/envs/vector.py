"""Ordered fixed-capacity vector adapter for live game workers."""

from __future__ import annotations

import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from autodancer.live.native_pipe import NativePipeError
from autodancer.live.protocol import ProtocolError
from autodancer.live.supervisor import AutoDancerSupervisor
from autodancer.memory import MapCapacityError


class VectorInfrastructureError(RuntimeError):
    def __init__(self, worker_index: int, worker_id: str, cause: BaseException) -> None:
        self.worker_index = worker_index
        self.worker_id = worker_id
        self.cause = cause
        super().__init__(f"Infrastructure failure in {worker_id}: {type(cause).__name__}: {cause}")


class AutoDancerVectorEnv:
    def __init__(self, supervisor: AutoDancerSupervisor) -> None:
        self.supervisor = supervisor
        self.worker_ids = supervisor.worker_ids
        self.environments = {
            worker_id: supervisor.environment(worker_id) for worker_id in self.worker_ids
        }
        self._executor = ThreadPoolExecutor(max_workers=len(self.worker_ids))
        self.infrastructure_events: list[dict[str, Any]] = []

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
        self,
        seeds: list[int] | np.ndarray,
        options: list[dict[str, Any] | None] | None = None,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} seeds, received {len(seeds)}")
        reset_options = [None] * self.num_envs if options is None else options
        if len(reset_options) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} reset options, received {len(reset_options)}"
            )
        futures = [
            self._executor.submit(
                self.environments[worker_id].reset,
                seed=int(seed),
                options=worker_options,
            )
            for worker_id, seed, worker_options in zip(
                self.worker_ids, seeds, reset_options, strict=True
            )
        ]
        results = []
        for index, future in enumerate(futures):
            try:
                results.append(future.result())
            except MapCapacityError:
                raise
            except (TimeoutError, NativePipeError, ProtocolError) as error:
                results.append(
                    self.recover(
                        index,
                        int(seeds[index]),
                        options=reset_options[index],
                        failure=self._failure(index, error, operation="reset"),
                    )
                )
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
            except (TimeoutError, NativePipeError, ProtocolError) as error:
                self.recover(
                    index,
                    secrets.randbelow(2**31),
                    failure=self._failure(
                        index,
                        error,
                        operation="step",
                        action=int(actions[index]),
                    ),
                )
                raise VectorInfrastructureError(index, self.worker_ids[index], error) from error
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
        self,
        indices: list[int],
        seeds: list[int],
        options: list[dict[str, Any] | None] | None = None,
    ) -> list[tuple[dict[str, np.ndarray], dict[str, Any]]]:
        if len(indices) != len(seeds):
            raise ValueError("indices and seeds must have the same length")
        reset_options = [None] * len(indices) if options is None else options
        if len(reset_options) != len(indices):
            raise ValueError("indices and reset options must have the same length")
        futures = [
            self._executor.submit(
                self.environments[self.worker_ids[index]].reset,
                seed=int(seed),
                options=worker_options,
            )
            for index, seed, worker_options in zip(
                indices, seeds, reset_options, strict=True
            )
        ]
        results = []
        for index, seed, worker_options, future in zip(
            indices, seeds, reset_options, futures, strict=True
        ):
            try:
                results.append(future.result())
            except MapCapacityError:
                raise
            except (TimeoutError, NativePipeError, ProtocolError) as error:
                results.append(
                    self.recover(
                        index,
                        int(seed),
                        options=worker_options,
                        failure=self._failure(index, error, operation="reset_at"),
                    )
                )
        return results

    def _failure(
        self,
        worker_index: int,
        error: BaseException,
        *,
        operation: str,
        action: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        value = {
            "worker_index": worker_index,
            "worker_id": self.worker_ids[worker_index],
            "operation": operation,
            "error_type": type(error).__name__,
            "error": str(error),
            "action": action,
            **(context or {}),
        }
        self.infrastructure_events.append(value)
        return value

    def recover(
        self,
        worker_index: int,
        seed: int,
        *,
        options: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        worker_id = self.worker_ids[worker_index]
        self.supervisor.replace_worker(worker_id, failure=failure)
        self.environments[worker_id].close()
        environment = self.supervisor.environment(worker_id)
        self.environments[worker_id] = environment
        return environment.reset(seed=seed, options=options)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        for environment in self.environments.values():
            environment.close()
        self.supervisor.close()
