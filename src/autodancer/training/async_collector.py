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

from autodancer.curriculum import (
    EpisodeCurriculumSchedule,
    EpisodeResetSpec,
    WeightedResetSpec,
    fixed_reset_spec,
)
from autodancer.live.native_pipe import NativePipeError
from autodancer.live.protocol import ProtocolError
from autodancer.progress import deeper_level
from autodancer.rewards import RewardConfig, RewardTracker
from autodancer.training.action_contract import ActionContractMemory
from autodancer.training.model import START_ACTION, PolicyModel
from autodancer.training.natural_prefix import (
    DeathMetalPhaseTracker,
    NaturalPrefixConfig,
    NaturalPrefixError,
    natural_prefix_policy_sample,
)
from autodancer.training.ppo import RolloutBatch
from autodancer.training.seed_schedule import TrainingSeedSchedule


@dataclass(slots=True)
class ActorState:
    observation: dict[str, np.ndarray]
    info: dict[str, Any]
    hidden: Tensor
    reset_spec: EpisodeResetSpec = field(
        default_factory=lambda: EpisodeResetSpec("fixed", 1, None, "normal")
    )
    previous_action: int = START_ACTION
    previous_reward: float = 0.0
    episode_start: bool = True
    episode_return: float = 0.0
    episode_extrinsic_return: float = 0.0
    episode_shaping_return: float = 0.0
    episode_events: list[dict[str, Any]] = field(default_factory=list)
    episode_actions: list[int] = field(default_factory=list)
    furthest_zone: int = 0
    furthest_floor: int = 0
    boss_type: int = 0
    boss_progress: DeathMetalPhaseTracker = field(
        default_factory=lambda: DeathMetalPhaseTracker(NaturalPrefixConfig())
    )
    prefix_pending: bool = False
    prefix_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.furthest_zone, self.furthest_floor = deeper_level(
            (self.furthest_zone, self.furthest_floor),
            (int(self.info.get("zone") or 0), int(self.info.get("floor") or 0)),
        )
        self.boss_type = max(self.boss_type, int(self.info.get("boss_type") or 0))
        self.boss_progress.observe(self.observation, self.info)


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
                            (probabilities.cumsum(dim=-1) < samples.unsqueeze(-1))
                            .sum(dim=-1)
                            .clamp_max(probabilities.shape[-1] - 1)
                        )
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
        training_seed_pool: tuple[int, ...] = (),
        seed_schedule_state: dict[str, Any] | None = None,
        curriculum_entries: tuple[WeightedResetSpec, ...] = (),
        curriculum_schedule_state: dict[str, Any] | None = None,
        guide_model: PolicyModel | None = None,
        natural_prefix: NaturalPrefixConfig | None = None,
        guide_reward_config: RewardConfig | None = None,
        policy_feedback_config: RewardConfig | None = None,
    ) -> None:
        self.environment = environment
        self.model = model
        self.device = device
        self.batch_delay = batch_delay
        self.telemetry_callback = telemetry_callback
        prefix_parts = (guide_model, natural_prefix, guide_reward_config)
        if not (
            all(value is None for value in prefix_parts)
            or all(value is not None for value in prefix_parts)
        ):
            raise ValueError(
                "guide_model, natural_prefix, and guide_reward_config must be supplied together"
            )
        self.guide_model = guide_model
        self.natural_prefix = natural_prefix
        self.guide_reward_config = guide_reward_config
        self.policy_feedback_config = policy_feedback_config
        self.action_contract = action_contract
        self.contract_memory = ActionContractMemory(action_contract, environment.num_envs)
        self.base_seed = int(seed)
        self.seed_schedule = TrainingSeedSchedule(
            int(seed), environment.num_envs, tuple(training_seed_pool)
        )
        if seed_schedule_state is not None:
            self.seed_schedule.load_state_dict(seed_schedule_state)
        selected_entries = curriculum_entries or fixed_reset_spec(1, None, "normal")
        self.curriculum_schedule = EpisodeCurriculumSchedule(
            int(seed), environment.num_envs, tuple(selected_entries)
        )
        if curriculum_schedule_state is not None:
            self.curriculum_schedule.load_state_dict(curriculum_schedule_state)
        initial_resets = [self._next_reset(i) for i in range(environment.num_envs)]
        observations, infos = environment.reset(
            [item[0] for item in initial_resets],
            options=[item[1].reset_options() for item in initial_resets],
        )
        if self.natural_prefix is None:
            observations = self.contract_memory.reset_batch(observations)
        initial_hidden = model.initial_state(environment.num_envs, device=device)
        self.states = [
            ActorState(
                {key: value[index].copy() for key, value in observations.items()},
                infos[index],
                initial_hidden[index : index + 1],
                initial_resets[index][1],
                prefix_pending=self.natural_prefix is not None,
            )
            for index in range(environment.num_envs)
        ]
        self._policy_feedback_trackers = [
            self._new_policy_feedback_tracker(state.observation, state.info)
            for state in self.states
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
        self._guide_scheduler = (
            None
            if guide_model is None
            else InferenceScheduler(
                guide_model,
                device=device,
                max_batch=environment.num_envs,
                batch_delay=batch_delay,
                deterministic=bool(natural_prefix and natural_prefix.deterministic_guide),
            )
        )
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
        return self.seed_schedule.next(index)

    def _next_reset(self, index: int) -> tuple[int, EpisodeResetSpec]:
        return self._seed(index), self.curriculum_schedule.next(index)

    def _new_policy_feedback_tracker(
        self,
        observation: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> RewardTracker | None:
        if self.policy_feedback_config is None:
            return None
        tracker = RewardTracker(self.policy_feedback_config)
        tracker.reset(observation, info)
        return tracker

    def _reset_policy_feedback(
        self,
        index: int,
        observation: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> None:
        self._policy_feedback_trackers[index] = self._new_policy_feedback_tracker(observation, info)

    def _policy_feedback(
        self,
        index: int,
        observation: dict[str, np.ndarray],
        info: dict[str, Any],
        *,
        terminated: bool,
        truncated: bool,
        fallback: float,
    ) -> float:
        tracker = self._policy_feedback_trackers[index]
        if tracker is None:
            return float(fallback)
        feedback, _ = tracker.score(
            observation,
            info,
            info.get("raw_events", ()),
            terminated=terminated,
            truncated=truncated,
        )
        return float(feedback)

    def seed_schedule_state(self) -> dict[str, Any]:
        return self.seed_schedule.state_dict()

    def curriculum_schedule_state(self) -> dict[str, Any]:
        return self.curriculum_schedule.state_dict()

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
            "wall_attempts": float(
                sum(fragment.metrics["wall_attempts"] for fragment in fragments)
            ),
            "known_invalid_wall_discoveries": float(
                sum(fragment.metrics["known_invalid_wall_discoveries"] for fragment in fragments)
            ),
            "mean_masked_directions": float(
                np.mean(
                    [
                        fragment.metrics["masked_direction_observations"] / length
                        for fragment in fragments
                    ]
                )
            ),
            "mean_effective_masked_directions": float(
                np.mean(
                    [
                        fragment.metrics["effective_masked_direction_observations"] / length
                        for fragment in fragments
                    ]
                )
            ),
            "navigation_prior_rate": float(
                np.mean(
                    [fragment.metrics["navigation_prior_turns"] / length for fragment in fragments]
                )
            ),
            "max_remembered_hazards": float(
                max(fragment.metrics["max_remembered_hazards"] for fragment in fragments)
            ),
            "natural_prefix_failures": float(
                sum(fragment.metrics["natural_prefix_failures"] for fragment in fragments)
            ),
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
                    index,
                    int(state.info.get("seed", 0)),
                    options=state.reset_spec.reset_options(),
                    failure=failure,
                )
                if self.natural_prefix is None:
                    observation = self.contract_memory.reset_slot(index, observation)
                self.states[index] = ActorState(
                    observation,
                    info,
                    self.model.initial_state(1, device=self.device),
                    state.reset_spec,
                    prefix_pending=self.natural_prefix is not None,
                )
                self._reset_policy_feedback(index, observation, info)
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
        wall_attempts = 0
        known_invalid_wall_discoveries = 0
        masked_direction_observations = 0
        effective_masked_direction_observations = 0
        navigation_prior_turns = 0
        max_remembered_hazards = 0
        natural_prefix_failures = 0
        started = time.monotonic()
        for fragment_step in range(length):
            while state.prefix_pending:
                try:
                    state = self._run_natural_prefix(index, state, scheduler)
                except NaturalPrefixError as error:
                    natural_prefix_failures += 1
                    metadata = {
                        **error.config.specification(),
                        "acquired": False,
                        "attempts": len(error.failures),
                        "guide_turns": error.guide_turns,
                        "failures": error.failures,
                        "boundary": error.failures[-1].get("boundary", {}),
                    }
                    episodes.append(
                        {
                            "worker_id": worker_id,
                            "run_id": str(state.info.get("run_id", "")),
                            "seed": int(state.info.get("seed", 0)),
                            "return": 0.0,
                            "extrinsic_return": 0.0,
                            "shaping_return": 0.0,
                            "status": "prefix_failed",
                            "zone": state.furthest_zone,
                            "floor": state.furthest_floor,
                            "boss_type": state.boss_type,
                            "boss_progress": state.boss_progress.snapshot(),
                            "turns": 0,
                            "events": [],
                            "curriculum_reset": state.reset_spec.as_dict(),
                            "curriculum_reset_id": state.reset_spec.id,
                            "natural_prefix": metadata,
                            "infrastructure_valid": True,
                        }
                    )
                    self.curriculum_schedule.record_outcome(state.reset_spec, "prefix_failed")
                    if natural_prefix_failures >= error.config.max_failed_seeds_per_fragment:
                        raise NaturalPrefixError(
                            worker_id,
                            error.config,
                            failures=error.failures,
                            guide_turns=error.guide_turns,
                            observation=error.observation,
                            info={
                                **error.info,
                                "failure_reason": "prefix_failure_budget_exhausted",
                                "failed_seeds": natural_prefix_failures,
                            },
                        ) from error
                    next_seed, next_spec = self._next_reset(index)
                    next_observation, next_info = worker.reset(
                        seed=next_seed,
                        options=next_spec.reset_options(),
                    )
                    state = ActorState(
                        next_observation,
                        next_info,
                        self.model.initial_state(1, device=self.device),
                        next_spec,
                        prefix_pending=True,
                    )
                    self._publish_telemetry(index, next_observation, next_info, None, None)
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
            raw_next_observation, reward, terminated, truncated, info = worker.step(action)
            contract_diagnostic = self.contract_memory.observe(
                index,
                state.observation,
                action,
                raw_next_observation,
                info,
            )
            info["action_contract"] = contract_diagnostic
            next_observation = self.contract_memory.apply_slot(index, raw_next_observation)
            if str((info.get("action_outcome") or {}).get("category", "")) == "wall_attempt":
                wall_attempts += 1
            known_invalid_wall_discoveries += int(
                bool(contract_diagnostic["newly_learned_invalid_wall"])
            )
            masked_direction_observations += int(contract_diagnostic["masked_direction_count"])
            effective_masked_direction_observations += int(
                contract_diagnostic["effective_masked_direction_count"]
            )
            navigation_prior_turns += int(bool(contract_diagnostic["navigation_prior_active"]))
            max_remembered_hazards = max(
                max_remembered_hazards,
                int(contract_diagnostic.get("remembered_hazards", 0)),
            )
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
            policy_feedback = self._policy_feedback(
                index,
                next_observation,
                info,
                terminated=bool(terminated),
                truncated=bool(truncated),
                fallback=reward,
            )
            truncation_value = torch.tensor(0.0, dtype=torch.float32)
            if truncated and not terminated:
                # A client time limit is not an MDP terminal. Bootstrap the
                # critic from the real final observation, then stop the GAE
                # trace at the reset boundary.
                _, _, terminal_value, _ = scheduler.infer(
                    next_observation,
                    action,
                    policy_feedback,
                    next_hidden,
                    0.5,
                )
                truncation_value = terminal_value
            self._publish_telemetry(index, next_observation, info, action, reward)
            state.episode_return += reward
            state.episode_extrinsic_return += float(info.get("extrinsic_reward", 0.0))
            state.episode_shaping_return += float(info.get("shaping_reward", 0.0))
            state.episode_events.extend(info.get("raw_events", []))
            state.episode_actions.append(int(action))
            state.furthest_zone, state.furthest_floor = deeper_level(
                (state.furthest_zone, state.furthest_floor),
                (int(info.get("zone") or 0), int(info.get("floor") or 0)),
            )
            state.boss_type = max(state.boss_type, int(info.get("boss_type") or 0))
            state.boss_progress.observe(next_observation, info)
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
                        "run_id": str(info.get("run_id", state.info.get("run_id", ""))),
                        "seed": int(info.get("seed", state.info.get("seed", 0))),
                        "return": state.episode_return,
                        "extrinsic_return": state.episode_extrinsic_return,
                        "shaping_return": state.episode_shaping_return,
                        "status": info.get("episode_status"),
                        "zone": state.furthest_zone,
                        "floor": state.furthest_floor,
                        "boss_type": state.boss_type,
                        "boss_progress": state.boss_progress.snapshot(),
                        "turns": info.get("turns"),
                        "events": state.episode_events,
                        "actions": state.episode_actions,
                        "curriculum_reset": state.reset_spec.as_dict(),
                        "curriculum_reset_id": state.reset_spec.id,
                        "natural_prefix": dict(state.prefix_metadata),
                        "infrastructure_valid": True,
                    }
                )
                self.curriculum_schedule.record_outcome(
                    state.reset_spec, str(info.get("episode_status", "unknown"))
                )
                next_seed, next_spec = self._next_reset(index)
                next_observation, next_info = worker.reset(
                    seed=next_seed,
                    options=next_spec.reset_options(),
                )
                if self.natural_prefix is None:
                    next_observation = self.contract_memory.reset_slot(index, next_observation)
                state = ActorState(
                    next_observation,
                    next_info,
                    self.model.initial_state(1, device=self.device),
                    next_spec,
                    prefix_pending=self.natural_prefix is not None,
                )
                self._reset_policy_feedback(index, next_observation, next_info)
            else:
                state.observation = next_observation
                state.info = info
                state.hidden = next_hidden
                state.previous_action = action
                state.previous_reward = policy_feedback
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
                "wall_attempts": float(wall_attempts),
                "known_invalid_wall_discoveries": float(known_invalid_wall_discoveries),
                "masked_direction_observations": float(masked_direction_observations),
                "effective_masked_direction_observations": float(
                    effective_masked_direction_observations
                ),
                "navigation_prior_turns": float(navigation_prior_turns),
                "max_remembered_hazards": float(max_remembered_hazards),
                "natural_prefix_failures": float(natural_prefix_failures),
            },
        )

    def _run_natural_prefix(
        self,
        index: int,
        state: ActorState,
        learner_scheduler: InferenceScheduler,
    ) -> ActorState:
        """Run legal guide actions and return the exact reached learner state."""

        config = self.natural_prefix
        guide_scheduler = self._guide_scheduler
        if config is None or guide_scheduler is None or self.guide_model is None:
            return state
        worker_id = self.environment.worker_ids[index]
        worker = self.environment.environments[worker_id]
        seed = int(state.info.get("seed", 0))
        observation = self.contract_memory.reset_slot(index, state.observation)
        info = state.info
        total_guide_turns = 0
        failure_summaries: list[dict[str, Any]] = []

        for attempt in range(config.max_attempts):
            tracker = DeathMetalPhaseTracker(config)
            tracker.observe(observation, info)
            guide_hidden = self.guide_model.initial_state(1, device=self.device)
            learner_hidden = self.model.initial_state(1, device=self.device)
            previous_action = START_ACTION
            guide_previous_reward = 0.0
            learner_previous_reward = 0.0
            last_guide_action = START_ACTION
            last_learner_reward = 0.0
            if self.guide_reward_config is None:
                raise RuntimeError("Natural-prefix guide reward contract is missing")
            guide_reward_tracker = RewardTracker(self.guide_reward_config)
            guide_reward_tracker.reset(observation, info)
            learner_feedback_tracker = self._new_policy_feedback_tracker(observation, info)

            for guide_turn in range(config.max_guide_turns):
                sample = natural_prefix_policy_sample(
                    config.guide_policy_seed,
                    seed,
                    attempt,
                    guide_turn,
                )
                action, _, _, next_guide_hidden = guide_scheduler.infer(
                    observation,
                    previous_action,
                    guide_previous_reward,
                    guide_hidden,
                    sample,
                )
                next_learner_hidden = learner_hidden
                if config.recurrent_state_mode == "warm":
                    _, _, _, next_learner_hidden = learner_scheduler.infer(
                        observation,
                        previous_action,
                        learner_previous_reward,
                        learner_hidden,
                        0.5,
                    )
                raw_next_observation, reward, terminated, truncated, next_info = worker.step(action)
                total_guide_turns += 1
                next_info = dict(next_info)
                next_info["natural_prefix_stage"] = "guide"
                next_info["natural_prefix_attempt"] = attempt + 1
                next_info["natural_prefix_guide_turn"] = guide_turn + 1
                next_info["action_contract"] = self.contract_memory.observe(
                    index,
                    observation,
                    action,
                    raw_next_observation,
                    next_info,
                )
                next_observation = self.contract_memory.apply_slot(index, raw_next_observation)
                tracker.observe(next_observation, next_info)
                guide_reward, _ = guide_reward_tracker.score(
                    next_observation,
                    next_info,
                    next_info.get("raw_events", ()),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                if learner_feedback_tracker is None:
                    learner_feedback = float(reward)
                else:
                    learner_feedback, _ = learner_feedback_tracker.score(
                        next_observation,
                        next_info,
                        next_info.get("raw_events", ()),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                self._publish_telemetry(index, next_observation, next_info, action, float(reward))
                last_guide_action = action
                last_learner_reward = float(learner_feedback)

                if tracker.reached and not terminated and not truncated:
                    metadata = {
                        **config.specification(),
                        "acquired": True,
                        "attempts": attempt + 1,
                        "failed_attempts": len(failure_summaries),
                        "guide_turns": total_guide_turns,
                        "handoff_sequence": int(next_info.get("sequence", -1)),
                        "handoff_run_id": str(next_info.get("run_id", "")),
                        "handoff_seed": int(next_info.get("seed", seed)),
                        "boundary": tracker.snapshot(),
                        "guide_reward": self.guide_reward_config.specification(),
                    }
                    handed_observation, handed_info = worker.begin_learning_segment(
                        next_info,
                        metadata=metadata,
                    )
                    warm = config.recurrent_state_mode == "warm"
                    effective_observation = (
                        self.contract_memory.apply_slot(index, handed_observation)
                        if warm
                        else self.contract_memory.reset_slot(index, handed_observation)
                    )
                    self._policy_feedback_trackers[index] = learner_feedback_tracker
                    return ActorState(
                        effective_observation,
                        handed_info,
                        (
                            next_learner_hidden
                            if warm
                            else self.model.initial_state(1, device=self.device)
                        ),
                        state.reset_spec,
                        previous_action=last_guide_action if warm else START_ACTION,
                        previous_reward=last_learner_reward if warm else 0.0,
                        episode_start=not warm,
                        prefix_pending=False,
                        prefix_metadata=metadata,
                    )

                observation = next_observation
                info = next_info
                guide_hidden = next_guide_hidden
                learner_hidden = next_learner_hidden
                previous_action = action
                guide_previous_reward = float(guide_reward)
                learner_previous_reward = float(learner_feedback)
                if terminated or truncated:
                    break

            failure_summaries.append(
                {
                    "attempt": attempt + 1,
                    "turns": min(guide_turn + 1, config.max_guide_turns),
                    "status": str(info.get("episode_status", "guide_limit")),
                    "boundary": tracker.snapshot(),
                }
            )
            if attempt + 1 < config.max_attempts:
                observation, info = worker.reset(
                    seed=seed,
                    options=state.reset_spec.reset_options(),
                )
                observation = self.contract_memory.reset_slot(index, observation)

        raise NaturalPrefixError(
            worker_id,
            config,
            failures=failure_summaries,
            guide_turns=total_guide_turns,
            observation=observation,
            info=info,
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
        if self._guide_scheduler is not None:
            self._guide_scheduler.close()
