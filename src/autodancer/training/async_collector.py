"""Versioned asynchronous recurrent rollout collection for live workers."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from autodancer.live.native_pipe import NativePipeError
from autodancer.live.protocol import ProtocolError
from autodancer.training.action_contract import apply_action_contract
from autodancer.training.model import START_ACTION, PolicyModel
from autodancer.training.ppo import RolloutBatch


@dataclass(slots=True)
class ActorState:
    observation: dict[str, np.ndarray]
    info: dict[str, Any]
    hidden: Tensor
    previous_action: int = START_ACTION
    previous_reward: float = 0.0
    episode_start: bool = True
    episode_return: float = 0.0
    episode_extrinsic_return: float = 0.0
    episode_shaping_return: float = 0.0
    episode_events: list[dict[str, Any]] = field(default_factory=list)
    furthest_zone: int = 0
    furthest_floor: int = 0


@dataclass(slots=True)
class ActorFragment:
    observations: dict[str, Tensor]
    actions: Tensor
    old_log_probs: Tensor
    rewards: Tensor
    dones: Tensor
    terminations: Tensor
    truncation_values: Tensor
    episode_starts: Tensor
    values: Tensor
    hiddens: Tensor
    completed_episodes: list[dict[str, Any]]
    reward_components: dict[str, float]
    metrics: dict[str, float]


@dataclass(slots=True)
class _InferenceRequest:
    observation: dict[str, np.ndarray]
    previous_action: int
    previous_reward: float
    hidden: Tensor
    sample: float
    future: Future[tuple[int, Tensor, Tensor, Tensor]]


class InferenceScheduler:
    """Dynamically batch whichever actors are ready against one frozen policy."""

    def __init__(
        self,
        model: PolicyModel,
        *,
        device: torch.device,
        max_batch: int,
        batch_delay: float,
        deterministic: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.max_batch = max_batch
        self.batch_delay = batch_delay
        self.deterministic = deterministic
        self._queue: queue.Queue[_InferenceRequest | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="policy-inference", daemon=True)
        self._thread.start()

    def infer(
        self,
        observation: dict[str, np.ndarray],
        previous_action: int,
        previous_reward: float,
        hidden: Tensor,
        sample: float,
    ) -> tuple[int, Tensor, Tensor, Tensor]:
        future: Future[tuple[int, Tensor, Tensor, Tensor]] = Future()
        self._queue.put(
            _InferenceRequest(
                observation,
                previous_action,
                previous_reward,
                hidden.detach(),
                sample,
                future,
            )
        )
        return future.result()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        self.model.eval()
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.batch_delay
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    request = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    self._queue.put(None)
                    break
                batch.append(request)
            try:
                observation = {
                    key: torch.from_numpy(
                        np.stack([request.observation[key] for request in batch])
                    ).to(self.device)
                    for key in first.observation
                }
                observation["previous_action"] = torch.tensor(
                    [request.previous_action for request in batch],
                    dtype=torch.long,
                    device=self.device,
                )
                observation["previous_reward"] = torch.tensor(
                    [request.previous_reward for request in batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                hidden = torch.cat([request.hidden for request in batch]).to(self.device)
                with torch.inference_mode():
                    logits, values, next_hidden = self.model.step(observation, hidden)
                    distribution = Categorical(logits=logits)
                    probabilities = torch.softmax(logits, dim=-1)
                    if self.deterministic:
                        actions = logits.argmax(dim=-1)
                    else:
                        samples = torch.tensor(
                            [request.sample for request in batch],
                            dtype=probabilities.dtype,
                            device=self.device,
                        )
                        actions = (
                            probabilities.cumsum(dim=-1) < samples.unsqueeze(-1)
                        ).sum(dim=-1).clamp_max(probabilities.shape[-1] - 1)
                    log_probs = distribution.log_prob(actions)
                for index, request in enumerate(batch):
                    request.future.set_result(
                        (
                            int(actions[index].cpu()),
                            log_probs[index].detach().cpu(),
                            values[index].detach().cpu(),
                            next_hidden[index : index + 1].detach(),
                        )
                    )
            except BaseException as error:
                for request in batch:
                    request.future.set_exception(error)


class VersionedAsyncRolloutCollector:
    """Collect one independently timed, policy-versioned fragment per worker slot."""

    def __init__(
        self,
        environment: Any,
        model: PolicyModel,
        *,
        device: torch.device,
        seed: int,
        batch_delay: float = 0.002,
        telemetry_callback: Any | None = None,
        action_contract: str = "current",
        initial_policy_version: int = 0,
    ) -> None:
        self.environment = environment
        self.model = model
        self.device = device
        self.batch_delay = batch_delay
        self.telemetry_callback = telemetry_callback
        self.action_contract = action_contract
        self.base_seed = int(seed)
        seed_sequence = np.random.SeedSequence(seed)
        self.rngs = [
            np.random.default_rng(item) for item in seed_sequence.spawn(environment.num_envs)
        ]
        observations, infos = environment.reset(
            [self._seed(i) for i in range(environment.num_envs)]
        )
        observations = apply_action_contract(observations, self.action_contract)
        initial_hidden = model.initial_state(environment.num_envs, device=device)
        self.states = [
            ActorState(
                {key: value[index].copy() for key, value in observations.items()},
                infos[index],
                initial_hidden[index : index + 1],
            )
            for index in range(environment.num_envs)
        ]
        self.completed_episodes: list[dict[str, Any]] = []
        self.last_reward_components: dict[str, float] = {}
        self.last_runtime_metrics: dict[str, Any] = {}
        self.policy_version = int(initial_policy_version)
        self._recovery_lock = threading.Lock()
        self._recovery_counts: dict[str, int] = {}
        self._recovery_total = 0
        self._last_recovery_error = ""
        self._executor = ThreadPoolExecutor(max_workers=environment.num_envs)
        for index, state in enumerate(self.states):
            self._publish_telemetry(index, state.observation, state.info, None, None)

    def _publish_telemetry(
        self,
        index: int,
        observation: dict[str, np.ndarray],
        info: dict[str, Any],
        action: int | None,
        reward: float | None,
    ) -> None:
        if self.telemetry_callback is not None:
            self.telemetry_callback(index, observation, info, action, reward)

    def _seed(self, index: int) -> int:
        return int(self.rngs[index].integers(0, 2**31))

    def collect(self, length: int) -> RolloutBatch:
        with self._recovery_lock:
            recoveries_before = self._recovery_total
            counts_before = self._recovery_counts.copy()
        scheduler = InferenceScheduler(
            self.model,
            device=self.device,
            max_batch=self.environment.num_envs,
            batch_delay=self.batch_delay,
        )
        started = time.monotonic()
        futures = [
            self._executor.submit(self._collect_slot, index, length, scheduler)
            for index in range(self.environment.num_envs)
        ]
        try:
            fragments = [future.result() for future in futures]
        finally:
            scheduler.close()
        elapsed = time.monotonic() - started
        self.completed_episodes.extend(
            episode for fragment in fragments for episode in fragment.completed_episodes
        )
        components: dict[str, float] = {}
        for fragment in fragments:
            for name, value in fragment.reward_components.items():
                components[name] = components.get(name, 0.0) + value
        self.last_reward_components = components
        completion_times = [fragment.metrics["fragment_seconds"] for fragment in fragments]
        with self._recovery_lock:
            recovery_total = self._recovery_total
            recovery_counts = self._recovery_counts.copy()
            last_recovery_error = self._last_recovery_error
        self.last_runtime_metrics = {
            "policy_version": float(self.policy_version),
            "collector_seconds": elapsed,
            "collector_steps_per_second": length * len(fragments) / max(elapsed, 1e-9),
            "fragment_straggler_seconds": max(completion_times) - min(completion_times),
            "mean_inference_wait_seconds": float(
                np.mean([fragment.metrics["inference_wait_seconds"] for fragment in fragments])
            ),
            "mean_environment_wait_seconds": float(
                np.mean([fragment.metrics["environment_wait_seconds"] for fragment in fragments])
            ),
            "collector_recoveries": float(recovery_total - recoveries_before),
            "collector_recoveries_total": float(recovery_total),
            "last_recovery_error": last_recovery_error,
            **{
                f"collector_recovery_{name}": float(count - counts_before.get(name, 0))
                for name, count in recovery_counts.items()
            },
        }
        self.policy_version += 1
        next_value = self._bootstrap_values()
        fields = (
            "actions",
            "old_log_probs",
            "rewards",
            "dones",
            "terminations",
            "truncation_values",
            "episode_starts",
            "values",
            "hiddens",
        )
        stacked = {
            field: torch.stack([getattr(fragment, field) for fragment in fragments], dim=1)
            for field in fields
        }
        observation_keys = fragments[0].observations
        return RolloutBatch(
            observations={
                key: torch.stack([fragment.observations[key] for fragment in fragments], dim=1)
                for key in observation_keys
            },
            next_value=next_value,
            **stacked,
        )

    def _collect_slot(
        self, index: int, length: int, scheduler: InferenceScheduler
    ) -> ActorFragment:
        while True:
            try:
                return self._collect_slot_once(index, length, scheduler)
            except (TimeoutError, NativePipeError, ProtocolError) as error:
                reason = type(error).__name__.lower()
                with self._recovery_lock:
                    self._recovery_total += 1
                    self._recovery_counts[reason] = self._recovery_counts.get(reason, 0) + 1
                    self._last_recovery_error = f"{type(error).__name__}: {error}"
                state = self.states[index]
                failure = self.environment._failure(
                    index,
                    error,
                    operation="async_collect",
                    context={
                        "policy_version": self.policy_version,
                        "run_id": state.info.get("run_id"),
                        "seed": state.info.get("seed"),
                        "sequence": state.info.get("sequence"),
                        "previous_action": state.previous_action,
                        "observation_summary": {
                            "player": state.observation["player"].tolist(),
                            "legal_actions": np.flatnonzero(
                                state.observation["action_mask"]
                            ).tolist(),
                        },
                    },
                )
                observation, info = self.environment.recover(
                    index, self._seed(index), failure=failure
                )
                observation = apply_action_contract(observation, self.action_contract)
                self.states[index] = ActorState(
                    observation,
                    info,
                    self.model.initial_state(1, device=self.device),
                )
                self._publish_telemetry(index, observation, info, None, None)

    def _collect_slot_once(
        self, index: int, length: int, scheduler: InferenceScheduler
    ) -> ActorFragment:
        state = self.states[index]
        worker_id = self.environment.worker_ids[index]
        worker = self.environment.environments[worker_id]
        observations: dict[str, list[Tensor]] = {
            **{key: [] for key in state.observation},
            "previous_action": [],
            "previous_reward": [],
        }
        actions: list[Tensor] = []
        log_probs: list[Tensor] = []
        rewards: list[Tensor] = []
        dones: list[Tensor] = []
        terminations: list[Tensor] = []
        truncation_values: list[Tensor] = []
        starts: list[Tensor] = []
        values: list[Tensor] = []
        hiddens: list[Tensor] = []
        episodes: list[dict[str, Any]] = []
        components: dict[str, float] = {}
        inference_wait = 0.0
        environment_wait = 0.0
        started = time.monotonic()
        for fragment_step in range(length):
            for key, value in state.observation.items():
                observations[key].append(torch.from_numpy(value.copy()))
            observations["previous_action"].append(torch.tensor(state.previous_action))
            observations["previous_reward"].append(torch.tensor(state.previous_reward))
            starts.append(torch.tensor(state.episode_start))
            hiddens.append(state.hidden.squeeze(0).detach().cpu())
            inference_started = time.monotonic()
            action, log_prob, value, next_hidden = scheduler.infer(
                state.observation,
                state.previous_action,
                state.previous_reward,
                state.hidden,
                float(
                    np.random.default_rng(
                        np.random.SeedSequence(
                            [self.base_seed, index, self.policy_version, fragment_step]
                        )
                    ).random()
                ),
            )
            inference_wait += time.monotonic() - inference_started
            environment_started = time.monotonic()
            next_observation, reward, terminated, truncated, info = worker.step(action)
            next_observation = apply_action_contract(next_observation, self.action_contract)
            step_latency = time.monotonic() - environment_started
            environment_wait += step_latency
            supervisor = getattr(self.environment, "supervisor", None)
            if supervisor is not None:
                handle = supervisor.workers[worker_id]
                handle.last_latency = step_latency
                handle.episode_status = str(info.get("episode_status", "running"))
                acknowledgement = info.get("bridge") or {}
                handle.last_acknowledged_command = int(
                    acknowledgement.get("command_id", handle.last_acknowledged_command)
                )
                handle.max_frame_bytes = max(
                    handle.max_frame_bytes, int(info.get("max_frame_bytes", 0))
                )
            done = bool(terminated or truncated)
            reward = float(reward)
            truncation_value = torch.tensor(0.0, dtype=torch.float32)
            if truncated and not terminated:
                # A client time limit is not an MDP terminal. Bootstrap the
                # critic from the real final observation, then stop the GAE
                # trace at the reset boundary.
                _, _, terminal_value, _ = scheduler.infer(
                    next_observation,
                    action,
                    reward,
                    next_hidden,
                    0.5,
                )
                truncation_value = terminal_value
            self._publish_telemetry(index, next_observation, info, action, reward)
            state.episode_return += reward
            state.episode_extrinsic_return += float(info.get("extrinsic_reward", 0.0))
            state.episode_shaping_return += float(info.get("shaping_reward", 0.0))
            state.episode_events.extend(info.get("raw_events", []))
            state.furthest_zone = max(state.furthest_zone, int(info.get("zone") or 0))
            state.furthest_floor = max(state.furthest_floor, int(info.get("floor") or 0))
            for name, component in info.get("reward_components", {}).items():
                components[name] = components.get(name, 0.0) + float(component)
            actions.append(torch.tensor(action, dtype=torch.long))
            log_probs.append(log_prob)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            dones.append(torch.tensor(done))
            terminations.append(torch.tensor(bool(terminated)))
            truncation_values.append(truncation_value)
            values.append(value)
            if done:
                episodes.append(
                    {
                        "worker_id": worker_id,
                        "return": state.episode_return,
                        "extrinsic_return": state.episode_extrinsic_return,
                        "shaping_return": state.episode_shaping_return,
                        "status": info.get("episode_status"),
                        "zone": state.furthest_zone,
                        "floor": state.furthest_floor,
                        "turns": info.get("turns"),
                        "events": state.episode_events,
                    }
                )
                next_observation, next_info = worker.reset(seed=self._seed(index))
                next_observation = apply_action_contract(
                    next_observation, self.action_contract
                )
                state = ActorState(
                    next_observation,
                    next_info,
                    self.model.initial_state(1, device=self.device),
                )
            else:
                state.observation = next_observation
                state.info = info
                state.hidden = next_hidden
                state.previous_action = action
                state.previous_reward = reward
                state.episode_start = False
        self.states[index] = state
        return ActorFragment(
            observations={key: torch.stack(value) for key, value in observations.items()},
            actions=torch.stack(actions),
            old_log_probs=torch.stack(log_probs),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            terminations=torch.stack(terminations),
            truncation_values=torch.stack(truncation_values),
            episode_starts=torch.stack(starts),
            values=torch.stack(values),
            hiddens=torch.stack(hiddens),
            completed_episodes=episodes,
            reward_components=components,
            metrics={
                "fragment_seconds": time.monotonic() - started,
                "inference_wait_seconds": inference_wait,
                "environment_wait_seconds": environment_wait,
            },
        )

    def _bootstrap_values(self) -> Tensor:
        observation = {
            key: torch.from_numpy(np.stack([state.observation[key] for state in self.states])).to(
                self.device
            )
            for key in self.states[0].observation
        }
        observation["previous_action"] = torch.tensor(
            [state.previous_action for state in self.states], device=self.device
        )
        observation["previous_reward"] = torch.tensor(
            [state.previous_reward for state in self.states],
            dtype=torch.float32,
            device=self.device,
        )
        hidden = torch.cat([state.hidden for state in self.states]).to(self.device)
        with torch.inference_mode():
            _, values, _ = self.model.step(observation, hidden)
        return values.cpu()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
