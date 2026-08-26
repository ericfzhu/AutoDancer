from __future__ import annotations

import threading
import time

import numpy as np
import torch

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    MAP_CHANNELS,
    MAP_SIZE,
    PLAYER_FEATURES,
    Action,
    GridChannel,
    Terrain,
)
from autodancer.training.async_collector import VersionedAsyncRolloutCollector
from autodancer.training.model import ModelConfig, RecurrentActorCritic
from autodancer.training.seed_schedule import TrainingSeedSchedule


def observation(slot: int) -> dict[str, np.ndarray]:
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    mask[:4] = 1
    player = np.zeros(PLAYER_FEATURES, dtype=np.int32)
    player[0] = 6
    player[1] = 6
    player[6] = 1
    player[7] = 1
    player[8] = slot
    return {
        "grid": np.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16),
        "map_memory": np.zeros((MAP_SIZE, MAP_SIZE, MAP_CHANNELS), dtype=np.int16),
        "player": player,
        "inventory": np.zeros((INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16),
        "action_mask": mask,
    }


class DelayedWorker:
    def __init__(self, slot: int, delays: list[float], timeline: list[tuple], lock: threading.Lock):
        self.slot = slot
        self.delays = delays
        self.timeline = timeline
        self.lock = lock
        self.turn = 0

    def reset(self, *, seed: int):
        return observation(self.slot), {"seed": seed, "episode_status": "running"}

    def step(self, action: int):
        turn = self.turn
        self.turn += 1
        with self.lock:
            self.timeline.append(("start", self.slot, turn, time.monotonic()))
        time.sleep(self.delays[turn])
        with self.lock:
            self.timeline.append(("end", self.slot, turn, time.monotonic()))
        return (
            observation(self.slot),
            float(action) / 100,
            False,
            False,
            {"episode_status": "running", "zone": 1, "floor": 1},
        )


class AsyncEnvironment:
    num_envs = 2
    worker_ids = ["worker-0000", "worker-0001"]

    def __init__(self) -> None:
        self.timeline: list[tuple] = []
        lock = threading.Lock()
        self.environments = {
            "worker-0000": DelayedWorker(0, [0.0, 0.0, 0.0], self.timeline, lock),
            "worker-0001": DelayedWorker(1, [0.05, 0.0, 0.0], self.timeline, lock),
        }
        self.infrastructure_events: list[dict] = []

    def reset(self, seeds: list[int]):
        results = [
            self.environments[worker_id].reset(seed=seed)
            for worker_id, seed in zip(self.worker_ids, seeds, strict=True)
        ]
        return (
            {key: np.stack([result[0][key] for result in results]) for key in results[0][0]},
            [result[1] for result in results],
        )

    def _failure(self, index: int, error: BaseException, **details):
        value = {"index": index, "error": str(error), **details}
        self.infrastructure_events.append(value)
        return value

    def recover(self, index: int, seed: int, *, failure=None):
        del failure
        return self.environments[self.worker_ids[index]].reset(seed=seed)


def test_async_collector_uses_and_resumes_finite_seed_pool() -> None:
    model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    pool = (101, 102, 103)
    first = VersionedAsyncRolloutCollector(
        AsyncEnvironment(),
        model,
        device=torch.device("cpu"),
        seed=77,
        training_seed_pool=pool,
    )
    try:
        assert {int(state.info["seed"]) for state in first.states} <= set(pool)
        saved = first.seed_schedule_state()
    finally:
        first.close()

    expected_schedule = TrainingSeedSchedule(77, 2, pool)
    expected_schedule.load_state_dict(saved)
    expected = [expected_schedule.next(0), expected_schedule.next(1)]
    resumed = VersionedAsyncRolloutCollector(
        AsyncEnvironment(),
        model,
        device=torch.device("cpu"),
        seed=77,
        training_seed_pool=pool,
        seed_schedule_state=saved,
    )
    try:
        assert [int(state.info["seed"]) for state in resumed.states] == expected
    finally:
        resumed.close()


def test_versioned_async_collection_has_no_per_step_worker_barrier() -> None:
    environment = AsyncEnvironment()
    # Leave enough separation for this scheduling assertion on a machine that is
    # simultaneously running CPU-heavy live game workers.
    environment.environments["worker-0001"].delays[0] = 0.5
    model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    collector = VersionedAsyncRolloutCollector(
        environment, model, device=torch.device("cpu"), seed=3, batch_delay=0.001
    )
    try:
        rollout = collector.collect(3)
    finally:
        collector.close()
    assert rollout.actions.shape == (3, 2)
    assert rollout.hiddens.shape[:3] == (3, 2, 2)
    start_worker0_turn1 = next(
        item[3] for item in environment.timeline if item[:3] == ("start", 0, 1)
    )
    end_worker1_turn0 = next(item[3] for item in environment.timeline if item[:3] == ("end", 1, 0))
    assert start_worker0_turn1 < end_worker1_turn0
    assert collector.last_runtime_metrics["policy_version"] == 0


def test_async_collector_publishes_each_worker_turn_live() -> None:
    environment = AsyncEnvironment()
    model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    updates: list[tuple[int, int | None, float | None]] = []
    collector = VersionedAsyncRolloutCollector(
        environment,
        model,
        device=torch.device("cpu"),
        seed=3,
        batch_delay=0.001,
        telemetry_callback=lambda index, _observation, _info, action, reward: updates.append(
            (index, action, reward)
        ),
    )
    try:
        collector.collect(3)
    finally:
        collector.close()
    assert len(updates) == 2 + (2 * 3)
    assert updates[:2] == [(0, None, None), (1, None, None)]
    assert all(action is not None and reward is not None for _, action, reward in updates[2:])


def test_async_collector_applies_episode_local_known_wall_memory() -> None:
    environment = AsyncEnvironment()

    def wall_observation(slot: int) -> dict[str, np.ndarray]:
        result = observation(slot)
        result["grid"][
            GRID_SIZE // 2,
            GRID_SIZE // 2 + 1,
            GridChannel.TERRAIN_CLASS,
        ] = Terrain.WALL
        return result

    seen_actions: dict[int, list[int]] = {0: [], 1: []}
    for slot, worker_id in enumerate(environment.worker_ids):
        worker = environment.environments[worker_id]
        worker.reset = (  # type: ignore[method-assign]
            lambda *, seed, slot=slot: (
                wall_observation(slot),
                {"seed": seed, "episode_status": "running"},
            )
        )

        def step(action: int, *, slot: int = slot):
            seen_actions[slot].append(action)
            category = "wall_attempt" if action == int(Action.RIGHT) else "move"
            return (
                wall_observation(slot),
                0.0,
                False,
                False,
                {
                    "episode_status": "running",
                    "zone": 1,
                    "floor": 1,
                    "action_outcome": {"category": category},
                },
            )

        worker.step = step  # type: ignore[method-assign]

    model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.actor[-1].bias[int(Action.RIGHT)] = 20.0
    collector = VersionedAsyncRolloutCollector(
        environment,
        model,
        device=torch.device("cpu"),
        seed=8,
        batch_delay=0.001,
        action_contract="known-invalid-wall-v1",
    )
    try:
        rollout = collector.collect(2)
    finally:
        collector.close()
    assert np.all(rollout.actions[0].numpy() == int(Action.RIGHT))
    assert np.all(rollout.actions[1].numpy() != int(Action.RIGHT))
    assert collector.last_runtime_metrics["wall_attempts"] == 2
    assert collector.last_runtime_metrics["known_invalid_wall_discoveries"] == 2
    assert collector.last_runtime_metrics["mean_masked_directions"] == 1


def test_async_collector_bootstraps_truncation_from_terminal_observation() -> None:
    environment = AsyncEnvironment()
    worker = environment.environments["worker-0000"]

    def truncate(_action: int):
        return (
            observation(0),
            0.25,
            False,
            True,
            {"episode_status": "time_limit", "zone": 1, "floor": 1},
        )

    worker.step = truncate  # type: ignore[method-assign]
    model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.critic[-1].bias.fill_(2.0)
    collector = VersionedAsyncRolloutCollector(
        environment, model, device=torch.device("cpu"), seed=5, batch_delay=0.001
    )
    try:
        rollout = collector.collect(1)
    finally:
        collector.close()
    assert rollout.dones[0, 0]
    assert not rollout.terminations[0, 0]
    assert rollout.truncation_values[0, 0] == 2.0
    assert rollout.truncation_values[0, 1] == 0.0
    assert collector.states[0].episode_start


def test_async_collector_discards_only_failed_slot_fragment() -> None:
    environment = AsyncEnvironment()
    original = environment.environments["worker-0001"]
    original_step = original.step
    failed = False
    recoveries: list[int] = []

    def fail_once(action: int):
        nonlocal failed
        if not failed:
            failed = True
            raise TimeoutError("worker disconnected")
        return original_step(action)

    original.step = fail_once  # type: ignore[method-assign]
    original_recover = environment.recover

    def recover(index: int, seed: int, *, failure=None):
        del failure
        recoveries.append(index)
        return original_recover(index, seed)

    environment.recover = recover  # type: ignore[method-assign]
    model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    collector = VersionedAsyncRolloutCollector(
        environment, model, device=torch.device("cpu"), seed=4, batch_delay=0.001
    )
    try:
        rollout = collector.collect(2)
    finally:
        collector.close()
    assert rollout.actions.shape == (2, 2)
    assert recoveries == [1]
    assert collector.states[1].previous_action != -1
    assert collector.last_runtime_metrics["collector_recoveries"] == 1
    assert collector.last_runtime_metrics["collector_recovery_timeouterror"] == 1
    assert "worker disconnected" in collector.last_runtime_metrics["last_recovery_error"]


def test_stochastic_actions_are_independent_of_worker_timing() -> None:
    torch.manual_seed(17)
    first_model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    second_model = RecurrentActorCritic(first_model.config)
    second_model.load_state_dict(first_model.state_dict())
    first_environment = AsyncEnvironment()
    second_environment = AsyncEnvironment()
    first_environment.environments["worker-0000"].delays = [0.05, 0.0, 0.0]
    first_environment.environments["worker-0001"].delays = [0.0, 0.0, 0.0]
    second_environment.environments["worker-0000"].delays = [0.0, 0.0, 0.0]
    second_environment.environments["worker-0001"].delays = [0.05, 0.0, 0.0]
    first = VersionedAsyncRolloutCollector(
        first_environment,
        first_model,
        device=torch.device("cpu"),
        seed=91,
        batch_delay=0.001,
        initial_policy_version=7,
    )
    second = VersionedAsyncRolloutCollector(
        second_environment,
        second_model,
        device=torch.device("cpu"),
        seed=91,
        batch_delay=0.001,
        initial_policy_version=7,
    )
    try:
        first_rollout = first.collect(3)
        second_rollout = second.collect(3)
    finally:
        first.close()
        second.close()
    np.testing.assert_array_equal(first_rollout.actions, second_rollout.actions)
