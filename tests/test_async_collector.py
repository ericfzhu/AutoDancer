from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from autodancer.adaptive_curriculum import AdaptiveCurriculumConfig
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
    ActorKind,
    BossType,
    GridChannel,
    MapChannel,
    PlayerFeature,
    Terrain,
)
from autodancer.curriculum import EpisodeResetSpec, WeightedResetSpec
from autodancer.rewards import RewardConfig
from autodancer.training.async_collector import ActorState, VersionedAsyncRolloutCollector
from autodancer.training.baseline import _evaluate_model_async
from autodancer.training.model import ModelConfig, RecurrentActorCritic
from autodancer.training.natural_prefix import NaturalPrefixConfig, NaturalPrefixError
from autodancer.training.seed_schedule import TrainingSeedSchedule


def test_actor_state_initializes_progress_from_reset_level() -> None:
    state = ActorState(
        observation(0),
        {"zone": 2, "floor": 1},
        torch.zeros(1, 2),
    )
    assert (state.furthest_zone, state.furthest_floor) == (2, 1)


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
        self.reset_calls: list[tuple[int, object]] = []

    def reset(self, *, seed: int, options=None):
        self.reset_calls.append((seed, options))
        return observation(self.slot), {
            "seed": seed,
            "episode_status": "running",
            "curriculum_reset": None if options is None else options.get("curriculum"),
        }

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

    def reset(self, seeds: list[int], options=None):
        reset_options = [None] * len(seeds) if options is None else options
        results = [
            self.environments[worker_id].reset(seed=seed, options=worker_options)
            for worker_id, seed, worker_options in zip(
                self.worker_ids, seeds, reset_options, strict=True
            )
        ]
        return (
            {key: np.stack([result[0][key] for result in results]) for key in results[0][0]},
            [result[1] for result in results],
        )

    def _failure(self, index: int, error: BaseException, **details):
        value = {"index": index, "error": str(error), **details}
        self.infrastructure_events.append(value)
        return value

    def recover(self, index: int, seed: int, *, options=None, failure=None):
        del failure
        return self.environments[self.worker_ids[index]].reset(seed=seed, options=options)


class PrefixWorker:
    def __init__(self, slot: int, *, direct_mutation: bool = False) -> None:
        self.slot = slot
        self.direct_mutation = direct_mutation
        self.turn = 0
        self.handoffs: list[dict] = []
        self.step_calls = 0
        self.actions: list[int] = []
        self.seed = 0
        self.run_id = ""

    def _observation(self, health: int, actor_type: int) -> dict[str, np.ndarray]:
        result = observation(self.slot)
        result["player"][PlayerFeature.TASK] = int(BossType.DEATH_METAL)
        result["grid"][10, 11, GridChannel.ACTOR_CLASS] = int(ActorKind.BOSS)
        result["grid"][10, 11, GridChannel.ACTOR_TYPE] = actor_type
        result["grid"][10, 11, GridChannel.HEALTH] = health
        result["grid"][10, 11, GridChannel.MAX_HEALTH] = 9
        result["grid"][10, 11, GridChannel.VISIBILITY] = 2
        return result

    def reset(self, *, seed: int, options=None):
        self.turn = 0
        self.seed = seed
        self.run_id = f"run-{self.slot}-{seed}"
        return self._observation(9, 101), {
            "seed": seed,
            "run_id": self.run_id,
            "sequence": 0,
            "episode_status": "running",
            "zone": 1,
            "floor": 4,
        }

    def step(self, action: int):
        self.actions.append(action)
        self.turn += 1
        self.step_calls += 1
        if self.direct_mutation:
            health, actor_type, amount = 1, 101, 0
        else:
            phases = [(6, 102, 3), (4, 103, 2), (2, 104, 2)]
            health, actor_type, amount = phases[min(self.turn - 1, 2)]
        events = (
            []
            if amount == 0
            else [
                {
                    "kind": "enemy_damage",
                    "amount": amount,
                    "data": {"boss": True},
                }
            ]
        )
        return (
            self._observation(health, actor_type),
            10.0 if self.turn <= 3 else 0.25,
            False,
            False,
            {
                "seed": self.seed,
                "run_id": self.run_id,
                "sequence": self.turn,
                "episode_status": "running",
                "zone": 1,
                "floor": 4,
                "raw_events": events,
                "action_outcome": {"category": "wall_attempt"},
            },
        )

    def begin_learning_segment(self, info, *, metadata):
        self.handoffs.append(dict(metadata))
        handed = dict(info)
        handed["turns"] = 0
        handed["learning_segment"] = dict(metadata)
        return self._observation(2, 104), handed


class PrefixEnvironment(AsyncEnvironment):
    def __init__(self, *, direct_mutation: bool = False) -> None:
        self.timeline = []
        self.environments = {
            worker_id: PrefixWorker(slot, direct_mutation=direct_mutation)
            for slot, worker_id in enumerate(self.worker_ids)
        }
        self.infrastructure_events = []


class RecoveringPrefixWorker(PrefixWorker):
    """Expose one unreachable seed, followed by ordinary legal boss phases."""

    def __init__(self, slot: int) -> None:
        super().__init__(slot)
        self.reset_count = 0

    def reset(self, *, seed: int, options=None):
        self.direct_mutation = self.reset_count == 0
        self.reset_count += 1
        return super().reset(seed=seed, options=options)


class RecoveringPrefixEnvironment(PrefixEnvironment):
    def __init__(self) -> None:
        self.timeline = []
        self.environments = {
            worker_id: RecoveringPrefixWorker(slot)
            for slot, worker_id in enumerate(self.worker_ids)
        }
        self.infrastructure_events = []


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


def test_async_collector_routes_weighted_episode_resets_and_records_outcomes() -> None:
    environment = AsyncEnvironment()
    entries = (
        WeightedResetSpec(EpisodeResetSpec("reduced", 4, 5, "player10"), 0.75),
        WeightedResetSpec(EpisodeResetSpec("replay", 4, 5, "player20"), 0.25),
    )
    for slot, worker_id in enumerate(environment.worker_ids):
        environment.environments[worker_id].step = (  # type: ignore[method-assign]
            lambda _action, slot=slot: (
                observation(slot),
                1.0,
                True,
                False,
                {
                    "episode_status": "curriculum_complete",
                    "zone": 2,
                    "floor": 1,
                },
            )
        )
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
        environment,
        model,
        device=torch.device("cpu"),
        seed=81,
        curriculum_entries=entries,
    )
    try:
        collector.collect(10)
        state = collector.curriculum_schedule_state()
    finally:
        collector.close()
    assert sum(state["selected"].values()) == 22  # two initial plus twenty replacements
    assert sum(sum(values.values()) for values in state["outcomes"].values()) == 20
    assert {episode["curriculum_reset_id"] for episode in collector.completed_episodes} <= {
        "reduced",
        "replay",
    }
    assert all(len(episode["actions"]) == 1 for episode in collector.completed_episodes)
    assert all(0 <= episode["actions"][0] < 11 for episode in collector.completed_episodes)
    for worker in environment.environments.values():
        assert all(
            call_options is not None and "curriculum" in call_options
            for _, call_options in worker.reset_calls
        )


def test_async_collector_uses_opt_in_adaptive_curriculum_and_checkpoints_it() -> None:
    environment = AsyncEnvironment()
    entries = (
        WeightedResetSpec(EpisodeResetSpec("easier", 4, 5, "player20"), 1.0),
        WeightedResetSpec(EpisodeResetSpec("harder", 4, 5, "player10"), 1.0),
    )
    for slot, worker_id in enumerate(environment.worker_ids):
        environment.environments[worker_id].step = (  # type: ignore[method-assign]
            lambda _action, slot=slot: (
                observation(slot),
                1.0,
                True,
                False,
                {"episode_status": "curriculum_complete", "zone": 2, "floor": 1},
            )
        )
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
    adaptive = AdaptiveCurriculumConfig(
        window_size=10,
        minimum_samples=5,
        promotion_lower_bound=0.20,
        demotion_upper_bound=0.10,
    )
    collector = VersionedAsyncRolloutCollector(
        environment,
        model,
        device=torch.device("cpu"),
        seed=82,
        curriculum_entries=entries,
        adaptive_curriculum_config=adaptive,
    )
    try:
        collector.collect(10)
        state = collector.curriculum_schedule_state()
        resumed = VersionedAsyncRolloutCollector(
            AsyncEnvironment(),
            model,
            device=torch.device("cpu"),
            seed=82,
            curriculum_entries=entries,
            curriculum_schedule_state=state,
            adaptive_curriculum_config=adaptive,
        )
        resumed.close()
    finally:
        collector.close()
    assert state["mode"] == "adaptive-competence-v1"
    assert state["active_index"] == 1
    assert state["mastered_indices"] == [0, 1]


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


def test_async_collector_separates_policy_feedback_from_ppo_reward() -> None:
    environment = AsyncEnvironment()
    for slot, worker_id in enumerate(environment.worker_ids):
        environment.environments[worker_id].step = (  # type: ignore[method-assign]
            lambda _action, slot=slot: (
                observation(slot),
                10.0,
                False,
                False,
                {
                    "episode_status": "running",
                    "zone": 1,
                    "floor": 4,
                    "raw_events": [
                        {
                            "kind": "enemy_damage",
                            "amount": 1,
                            "data": {"boss": True},
                        }
                    ],
                },
            )
        )
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
    feedback = RewardConfig(
        new_position=0.0,
        new_tile=0.0,
        enemy_damage=0.2,
        max_combat_reward_per_floor=5.0,
        combat_reward_scope="boss_only",
        player_damage=0.0,
        new_item_type=0.0,
        stair_potential_max=0.0,
        floor_complete=0.0,
        zone_complete=0.0,
        victory=0.0,
        death=0.0,
    )
    collector = VersionedAsyncRolloutCollector(
        environment,
        model,
        device=torch.device("cpu"),
        seed=107,
        policy_feedback_config=feedback,
    )
    try:
        rollout = collector.collect(2)
    finally:
        collector.close()

    assert torch.allclose(rollout.rewards, torch.full((2, 2), 10.0))
    assert torch.allclose(rollout.observations["previous_reward"][0], torch.zeros(2))
    assert torch.allclose(rollout.observations["previous_reward"][1], torch.full((2,), 0.2))


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
            lambda *, seed, options=None, slot=slot: (
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
    assert collector.last_runtime_metrics["mean_effective_masked_directions"] == 1
    assert collector.last_runtime_metrics["navigation_prior_rate"] == 0


def test_navigation_prior_mask_and_log_probability_are_collected_on_policy() -> None:
    environment = AsyncEnvironment()

    def guided_observation(slot: int) -> dict[str, np.ndarray]:
        result = observation(slot)
        centre = MAP_SIZE // 2
        # UP is the only least-visited direction, so the contract must constrain
        # sampling before inference rather than replace the sampled action later.
        result["map_memory"][centre - 1, centre, MapChannel.VISIT_COUNT] = 0
        result["map_memory"][centre, centre + 1, MapChannel.VISIT_COUNT] = 3
        result["map_memory"][centre + 1, centre, MapChannel.VISIT_COUNT] = 3
        result["map_memory"][centre, centre - 1, MapChannel.VISIT_COUNT] = 3
        return result

    for slot, worker_id in enumerate(environment.worker_ids):
        worker = environment.environments[worker_id]
        worker.reset = (  # type: ignore[method-assign]
            lambda *, seed, options=None, slot=slot: (
                guided_observation(slot),
                {"seed": seed, "episode_status": "running"},
            )
        )

        def step(_action: int, *, slot: int = slot):
            return (
                guided_observation(slot),
                0.0,
                False,
                False,
                {
                    "episode_status": "running",
                    "zone": 1,
                    "floor": 1,
                    "action_outcome": {"category": "move"},
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
        # The unconstrained policy overwhelmingly prefers RIGHT. If the prior
        # were a post-sampling override, the stored likelihood would expose it.
        model.actor[-1].bias[int(Action.RIGHT)] = 20.0
    collector = VersionedAsyncRolloutCollector(
        environment,
        model,
        device=torch.device("cpu"),
        seed=9,
        batch_delay=0.001,
        action_contract="map-navigation-prior-v1",
    )
    try:
        rollout = collector.collect(1)
    finally:
        collector.close()

    assert torch.all(rollout.actions == int(Action.UP))
    assert torch.all(rollout.observations["action_mask"][..., int(Action.UP)] == 1)
    assert torch.all(rollout.observations["action_mask"][..., int(Action.RIGHT)] == 0)
    # Exactly one action is legal, hence its categorical probability is one.
    assert torch.allclose(rollout.old_log_probs, torch.zeros_like(rollout.old_log_probs))


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
    initial_seed = int(collector.states[0].info["seed"])
    try:
        rollout = collector.collect(1)
    finally:
        collector.close()
    assert rollout.dones[0, 0]
    assert not rollout.terminations[0, 0]
    assert rollout.truncation_values[0, 0] == 2.0
    assert rollout.truncation_values[0, 1] == 0.0
    assert collector.states[0].episode_start
    assert collector.completed_episodes[0]["seed"] == initial_seed


def test_episode_metrics_distinguish_curriculum_success_and_time_limits() -> None:
    from autodancer.training.train import episode_metrics

    metrics = episode_metrics(
        [
            {
                "seed": 41,
                "return": 10.0,
                "extrinsic_return": 10.0,
                "shaping_return": 0.0,
                "status": "curriculum_complete",
                "zone": 2,
                "floor": 1,
                "events": [],
            },
            {
                "seed": 42,
                "return": -1.0,
                "extrinsic_return": 0.0,
                "shaping_return": -1.0,
                "status": "time_limit",
                "zone": 1,
                "floor": 4,
                "events": [],
            },
        ]
    )

    assert metrics["completions"] == 0
    assert metrics["curriculum_completions"] == 1
    assert metrics["time_limits"] == 1
    assert metrics["episode_seeds"] == [41, 42]
    assert (metrics["furthest_zone"], metrics["furthest_floor"]) == (2, 1)
    assert metrics["furthest_level"] == 5


def test_async_collector_discards_only_failed_slot_fragment() -> None:
    environment = AsyncEnvironment()
    original = environment.environments["worker-0001"]
    original_step = original.step
    failed = False
    recoveries: list[tuple[int, int, object]] = []

    def fail_once(action: int):
        nonlocal failed
        if not failed:
            failed = True
            raise TimeoutError("worker disconnected")
        return original_step(action)

    original.step = fail_once  # type: ignore[method-assign]
    original_recover = environment.recover

    def recover(index: int, seed: int, *, options=None, failure=None):
        del failure
        recoveries.append((index, seed, options))
        return original_recover(index, seed, options=options)

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
    failed_seed = int(collector.states[1].info["seed"])
    failed_spec = collector.states[1].reset_spec.as_dict()
    try:
        rollout = collector.collect(2)
    finally:
        collector.close()
    assert rollout.actions.shape == (2, 2)
    assert recoveries == [(1, failed_seed, {"curriculum": failed_spec})]
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


def test_natural_prefix_uses_legal_guide_steps_but_excludes_them_from_rollout() -> None:
    environment = PrefixEnvironment()
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecurrentActorCritic(learner.config)
    collector = VersionedAsyncRolloutCollector(
        environment,
        learner,
        device=torch.device("cpu"),
        seed=100,
        guide_model=guide,
        guide_reward_config=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            target_phase=4,
            max_guide_turns=8,
            max_attempts=1,
            deterministic_guide=True,
            recurrent_state_mode="fresh",
        ),
    )
    try:
        rollout = collector.collect(2)
    finally:
        collector.close()

    assert rollout.actions.shape == (2, 2)
    assert torch.all(rollout.episode_starts[0])
    for worker in environment.environments.values():
        assert worker.step_calls == 5  # three guide turns plus two PPO turns
        assert len(worker.handoffs) == 1
        handoff = worker.handoffs[0]
        assert handoff["guide_transitions_in_ppo"] is False
        assert handoff["boundary"]["reached"] is True
        assert handoff["boundary"]["observed_actor_types"] == [101, 102, 103, 104]
    # The guide's deliberately large rewards are absent from learner returns.
    assert torch.allclose(rollout.rewards, torch.full((2, 2), 0.25))


def test_natural_prefix_guide_receives_its_own_reward_stream() -> None:
    class RecordingGuide(RecurrentActorCritic):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__(config)
            self.previous_rewards: list[float] = []

        def step(self, observation, state):
            self.previous_rewards.extend(
                float(value) for value in observation["previous_reward"].cpu()
            )
            return super().step(observation, state)

    environment = PrefixEnvironment()
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecordingGuide(learner.config)
    collector = VersionedAsyncRolloutCollector(
        environment,
        learner,
        device=torch.device("cpu"),
        seed=105,
        guide_model=guide,
        guide_reward_config=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            target_phase=4,
            max_guide_turns=8,
            max_attempts=1,
            deterministic_guide=True,
        ),
    )
    try:
        collector.collect(1)
    finally:
        collector.close()

    # PrefixWorker returns learner rewards of 10.0, while the checkpoint reward
    # contract scores the authoritative 3/2-point boss-damage events as .03/.02.
    assert max(guide.previous_rewards) < 1.0
    assert any(value == pytest.approx(0.03) for value in guide.previous_rewards)


def test_natural_prefix_warms_learner_with_stable_policy_feedback() -> None:
    class RecordingLearner(RecurrentActorCritic):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__(config)
            self.previous_rewards: list[float] = []

        def step(self, observation, state):
            self.previous_rewards.extend(
                float(value) for value in observation["previous_reward"].cpu()
            )
            return super().step(observation, state)

    environment = PrefixEnvironment()
    config = ModelConfig(
        cell_size=16,
        spatial_size=32,
        hidden_size=16,
        entity_limit=8,
        attention_layers=1,
        attention_heads=4,
    )
    learner = RecordingLearner(config)
    guide = RecurrentActorCritic(config)
    feedback = RewardConfig(
        new_position=0.0,
        new_tile=0.0,
        enemy_damage=0.2,
        max_combat_reward_per_floor=5.0,
        combat_reward_scope="boss_only",
        player_damage=0.0,
        new_item_type=0.0,
        stair_potential_max=0.0,
        floor_complete=0.0,
        zone_complete=0.0,
        victory=0.0,
        death=0.0,
    )
    collector = VersionedAsyncRolloutCollector(
        environment,
        learner,
        device=torch.device("cpu"),
        seed=108,
        guide_model=guide,
        guide_reward_config=RewardConfig(),
        policy_feedback_config=feedback,
        natural_prefix=NaturalPrefixConfig(
            target_phase=4,
            max_guide_turns=8,
            max_attempts=1,
            deterministic_guide=True,
            recurrent_state_mode="warm",
        ),
    )
    try:
        rollout = collector.collect(1)
    finally:
        collector.close()

    assert torch.allclose(rollout.rewards, torch.full((1, 2), 0.25))
    assert max(learner.previous_rewards) < 1.0
    assert any(value == pytest.approx(0.6) for value in learner.previous_rewards)
    assert any(value == pytest.approx(0.4) for value in learner.previous_rewards)


def test_natural_prefix_warm_mode_preserves_recurrent_context_at_handoff() -> None:
    environment = PrefixEnvironment()
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecurrentActorCritic(learner.config)
    collector = VersionedAsyncRolloutCollector(
        environment,
        learner,
        device=torch.device("cpu"),
        seed=101,
        guide_model=guide,
        guide_reward_config=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            max_guide_turns=8,
            max_attempts=1,
            recurrent_state_mode="warm",
        ),
    )
    try:
        rollout = collector.collect(1)
    finally:
        collector.close()
    assert not torch.any(rollout.episode_starts[0])
    assert torch.any(rollout.hiddens[0] != 0)


def test_natural_prefix_warm_handoff_preserves_action_contract_memory() -> None:
    environment = PrefixEnvironment()
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecurrentActorCritic(learner.config)
    with torch.no_grad():
        for parameter in guide.parameters():
            parameter.zero_()
        guide.actor[-1].bias[int(Action.UP)] = 20.0
    collector = VersionedAsyncRolloutCollector(
        environment,
        learner,
        device=torch.device("cpu"),
        seed=106,
        guide_model=guide,
        guide_reward_config=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            max_guide_turns=8,
            max_attempts=1,
            deterministic_guide=True,
            recurrent_state_mode="warm",
        ),
        action_contract="known-invalid-wall-v1",
    )
    try:
        collector.collect(1)
        assert all(collector.contract_memory._blocked[index] for index in range(2))
    finally:
        collector.close()


def test_natural_prefix_guide_uses_declared_stateful_action_contract() -> None:
    environment = PrefixEnvironment()
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecurrentActorCritic(learner.config)
    with torch.no_grad():
        for parameter in guide.parameters():
            parameter.zero_()
        guide.actor[-1].bias[int(Action.UP)] = 20.0
    collector = VersionedAsyncRolloutCollector(
        environment,
        learner,
        device=torch.device("cpu"),
        seed=104,
        guide_model=guide,
        guide_reward_config=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(max_guide_turns=3, max_attempts=1),
        action_contract="known-invalid-wall-v1",
    )
    try:
        collector.collect(1)
    finally:
        collector.close()

    for worker in environment.environments.values():
        assert worker.actions[0] == int(Action.UP)
        assert worker.actions[1] != int(Action.UP)


def test_natural_prefix_rejects_direct_boss_health_mutation() -> None:
    environment = PrefixEnvironment(direct_mutation=True)
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecurrentActorCritic(learner.config)
    collector = VersionedAsyncRolloutCollector(
        environment,
        learner,
        device=torch.device("cpu"),
        seed=102,
        guide_model=guide,
        guide_reward_config=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            max_guide_turns=3,
            max_attempts=1,
            max_failed_seeds_per_fragment=1,
        ),
    )
    try:
        with pytest.raises(NaturalPrefixError, match="failed to reach Death Metal phase 4"):
            collector.collect(1)
    finally:
        collector.close()


def test_natural_prefix_training_records_failed_seed_without_fabricating_transition() -> None:
    environment = RecoveringPrefixEnvironment()
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecurrentActorCritic(learner.config)
    collector = VersionedAsyncRolloutCollector(
        environment,
        learner,
        device=torch.device("cpu"),
        seed=103,
        guide_model=guide,
        guide_reward_config=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            max_guide_turns=3,
            max_attempts=1,
            max_failed_seeds_per_fragment=2,
        ),
    )
    initial_seeds = [int(state.info["seed"]) for state in collector.states]
    try:
        rollout = collector.collect(1)
    finally:
        collector.close()

    assert rollout.actions.shape == (1, 2)
    assert torch.allclose(rollout.rewards, torch.full((1, 2), 0.25))
    failures = [
        episode for episode in collector.completed_episodes if episode["status"] == "prefix_failed"
    ]
    assert [episode["seed"] for episode in failures] == initial_seeds
    assert all(episode["turns"] == 0 for episode in failures)
    assert all(episode["return"] == 0.0 for episode in failures)
    assert all(not episode["natural_prefix"]["acquired"] for episode in failures)
    assert collector.last_runtime_metrics["natural_prefix_failures"] == 2
    for worker in environment.environments.values():
        assert worker.step_calls == 7  # failed guide, legal guide, one learner turn
        assert len(worker.handoffs) == 1


def test_natural_prefix_evaluation_uses_same_reachable_handoff() -> None:
    environment = PrefixEnvironment()
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecurrentActorCritic(learner.config)
    results = _evaluate_model_async(
        environment,
        learner,
        seeds=[9001],
        max_steps=2,
        policy_seed=17,
        device=torch.device("cpu"),
        dashboard_state=None,
        action_contract="current",
        deterministic=True,
        guide_model=guide,
        guide_reward=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            max_guide_turns=8,
            max_attempts=1,
            deterministic_guide=True,
        ),
    )
    assert len(results) == 1
    assert results[0]["turns"] == 2
    assert environment.environments["worker-0000"].step_calls == 5
    assert len(environment.environments["worker-0000"].handoffs) == 1


def test_natural_prefix_evaluation_uses_checkpoint_guide_rewards() -> None:
    class RecordingGuide(RecurrentActorCritic):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__(config)
            self.previous_rewards: list[float] = []

        def step(self, observation, state):
            self.previous_rewards.extend(
                float(value) for value in observation["previous_reward"].cpu()
            )
            return super().step(observation, state)

    environment = PrefixEnvironment()
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecordingGuide(learner.config)
    _evaluate_model_async(
        environment,
        learner,
        seeds=[9004],
        max_steps=1,
        policy_seed=20,
        device=torch.device("cpu"),
        dashboard_state=None,
        action_contract="current",
        deterministic=True,
        guide_model=guide,
        guide_reward=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            max_guide_turns=8,
            max_attempts=1,
            deterministic_guide=True,
        ),
    )

    assert max(guide.previous_rewards) < 1.0
    assert any(value == pytest.approx(0.03) for value in guide.previous_rewards)


def test_natural_prefix_evaluation_uses_stable_learner_policy_feedback() -> None:
    class RecordingLearner(RecurrentActorCritic):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__(config)
            self.previous_rewards: list[float] = []

        def step(self, observation, state):
            self.previous_rewards.extend(
                float(value) for value in observation["previous_reward"].cpu()
            )
            return super().step(observation, state)

    environment = PrefixEnvironment()
    config = ModelConfig(
        cell_size=16,
        spatial_size=32,
        hidden_size=16,
        entity_limit=8,
        attention_layers=1,
        attention_heads=4,
    )
    learner = RecordingLearner(config)
    guide = RecurrentActorCritic(config)
    _evaluate_model_async(
        environment,
        learner,
        seeds=[9005],
        max_steps=2,
        policy_seed=21,
        device=torch.device("cpu"),
        dashboard_state=None,
        action_contract="current",
        deterministic=True,
        guide_model=guide,
        guide_reward=RewardConfig(),
        policy_feedback_config=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            max_guide_turns=8,
            max_attempts=1,
            deterministic_guide=True,
            recurrent_state_mode="warm",
        ),
    )

    assert learner.previous_rewards
    assert not any(value == pytest.approx(10.0) for value in learner.previous_rewards)
    assert not any(value == pytest.approx(0.25) for value in learner.previous_rewards)
    assert any(value == pytest.approx(0.02) for value in learner.previous_rewards)


def test_natural_prefix_evaluation_uses_declared_stateful_action_contract() -> None:
    environment = PrefixEnvironment()
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecurrentActorCritic(learner.config)
    with torch.no_grad():
        for parameter in guide.parameters():
            parameter.zero_()
        guide.actor[-1].bias[int(Action.UP)] = 20.0
    _evaluate_model_async(
        environment,
        learner,
        seeds=[9003],
        max_steps=1,
        policy_seed=19,
        device=torch.device("cpu"),
        dashboard_state=None,
        action_contract="known-invalid-wall-v1",
        deterministic=True,
        guide_model=guide,
        guide_reward=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(
            max_guide_turns=3,
            max_attempts=1,
            deterministic_guide=True,
        ),
    )
    worker = environment.environments["worker-0000"]
    assert worker.actions[0] == int(Action.UP)
    assert worker.actions[1] != int(Action.UP)


def test_natural_prefix_evaluation_reports_guide_failure_as_gameplay_outcome() -> None:
    environment = PrefixEnvironment(direct_mutation=True)
    learner = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            attention_heads=4,
        )
    )
    guide = RecurrentActorCritic(learner.config)
    results = _evaluate_model_async(
        environment,
        learner,
        seeds=[9002],
        max_steps=2,
        policy_seed=18,
        device=torch.device("cpu"),
        dashboard_state=None,
        action_contract="current",
        deterministic=True,
        guide_model=guide,
        guide_reward=RewardConfig(),
        natural_prefix=NaturalPrefixConfig(max_guide_turns=3, max_attempts=1),
    )
    assert len(results) == 1
    assert results[0]["status"] == "prefix_failed"
    assert results[0]["turns"] == 0
    assert results[0]["natural_prefix"]["acquired"] is False
    assert results[0]["natural_prefix"]["boundary"]["reached"] is False
    assert environment.infrastructure_events == []
