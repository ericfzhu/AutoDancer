from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

import autodancer.training.baseline as baseline_module
from autodancer.constants import (
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    Action,
    ActorKind,
    BossType,
    GridChannel,
    InventoryFeature,
    PlayerFeature,
    Terrain,
)
from autodancer.rewards import RewardConfig
from autodancer.training.baseline import (
    EpisodeAccumulator,
    compare_summaries,
    evaluate_live_policy,
    masked_random_actions,
    recurrent_state_after_transition,
    recurrent_state_for_action,
    resolve_policy_feedback_config,
    stochastic_policy_sample,
    summarize_episodes,
    validate_checkpoint_reward_schema,
    validate_checkpoint_trace_prefix,
    validate_declared_source_reference,
    zero_hidden_rows,
)


def test_policy_feedback_defaults_to_legacy_checkpoint_reward() -> None:
    trained = RewardConfig(profile_version=4, enemy_damage=0.123)
    evaluation = RewardConfig(
        profile_version=5,
        enemy_damage=0.0,
        boss_progress_potential_per_damage=0.2,
    )
    resolved, source, matches, checkpoint_spec = resolve_policy_feedback_config(
        {"checkpoint_metadata": {"reward": trained.specification()}},
        evaluation,
        None,
    )
    assert resolved == trained
    assert source == "checkpoint"
    assert matches is True
    assert checkpoint_spec == trained.specification()


def test_explicit_policy_feedback_override_is_reported_as_mismatch(tmp_path: Path) -> None:
    trained = RewardConfig(profile_version=4, enemy_damage=0.123)
    override = tmp_path / "feedback.json"
    override.write_text('{"profile_version": 4, "enemy_damage": 0.5}', encoding="utf-8")
    resolved, source, matches, checkpoint_spec = resolve_policy_feedback_config(
        {
            "checkpoint_metadata": {
                "reward": RewardConfig(profile_version=5).specification(),
                "policy_feedback_reward": trained.specification(),
            }
        },
        RewardConfig(profile_version=5),
        override,
    )
    assert resolved.enemy_damage == 0.5
    assert source == "explicit-override"
    assert matches is False
    assert checkpoint_spec == trained.specification()


def test_source_reference_requires_declared_path_and_hash(tmp_path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source-checkpoint")
    specification = {
        "source": {
            "checkpoint": "source.pt",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    }
    validate_declared_source_reference(specification, checkpoint, repository_root=tmp_path)

    specification["source"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        validate_declared_source_reference(specification, checkpoint, repository_root=tmp_path)


def test_custom_reward_label_still_checks_checkpoint_schema() -> None:
    checkpoint = {"checkpoint_metadata": {"reward": {"version": 4}}}
    validate_checkpoint_reward_schema(checkpoint, RewardConfig(profile_version=4))

    with pytest.raises(ValueError, match="reward schema"):
        validate_checkpoint_reward_schema(
            checkpoint,
            RewardConfig(
                profile_version=2,
                stair_potential_max=0.0,
                stair_potential_distance=0,
            ),
        )


def test_source_reference_can_use_common_evaluation_reward() -> None:
    checkpoint = {"checkpoint_metadata": {"reward": {"version": 4}}}

    validate_checkpoint_reward_schema(
        checkpoint,
        RewardConfig(profile_version=5),
        source_reference=True,
    )


def test_trace_prefix_evaluation_requires_exact_checkpoint_identity() -> None:
    class Prefix:
        def specification(self):
            return {
                "bank_sha256": "a" * 64,
                "qualification_sha256": "b" * 64,
                "action_contract": "current",
                "tail_actions": 16,
                "recurrent_state_mode": "warm",
            }

    prefix = Prefix()
    metadata = {"trace_prefix": prefix.specification()}
    validate_checkpoint_trace_prefix(metadata, prefix)  # type: ignore[arg-type]
    metadata["trace_prefix"] = {**prefix.specification(), "tail_actions": 32}
    with pytest.raises(ValueError, match="tail_actions"):
        validate_checkpoint_trace_prefix(metadata, prefix)  # type: ignore[arg-type]


def test_trace_prefix_source_reference_may_calibrate_a_new_tail_boundary() -> None:
    class Prefix:
        def specification(self):
            return {
                "bank_sha256": "a" * 64,
                "qualification_sha256": "b" * 64,
                "action_contract": "current",
                "tail_actions": 1,
                "recurrent_state_mode": "warm",
            }

    prefix = Prefix()
    validate_checkpoint_trace_prefix(
        {},
        prefix,
        source_reference=True,  # type: ignore[arg-type]
    )
    metadata = {"trace_prefix": {**prefix.specification(), "tail_actions": 16}}
    validate_checkpoint_trace_prefix(
        metadata,
        prefix,
        source_reference=True,  # type: ignore[arg-type]
    )
    metadata["trace_prefix"] = {
        **prefix.specification(),
        "bank_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="bank_sha256"):
        validate_checkpoint_trace_prefix(
            metadata,
            prefix,
            source_reference=True,  # type: ignore[arg-type]
        )


class OneStepEnvironment:
    num_envs = 4
    worker_ids = [f"worker-{index:04d}" for index in range(num_envs)]

    def __init__(self) -> None:
        self.reset_seeds: list[int] = []

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "player": np.zeros((self.num_envs, 16), dtype=np.int32),
            "action_mask": np.ones((self.num_envs, 11), dtype=np.int8),
        }

    def reset(self, seeds: list[int]):
        self.reset_seeds = seeds
        infos = [{"seed": seed, "run_id": str(seed), "zone": 1, "floor": 1} for seed in seeds]
        return self._observation(), infos

    def step(self, actions: np.ndarray):
        del actions
        infos = [
            {"episode_status": "dead", "zone": 1, "floor": 1, "raw_events": []}
            for _ in range(self.num_envs)
        ]
        return (
            self._observation(),
            np.zeros(self.num_envs, dtype=np.float32),
            np.ones(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            infos,
        )


def episode(seed: int, *, status: str, turns: int, kills: int = 0) -> dict[str, object]:
    return {
        "seed": seed,
        "worker_id": "worker-0000",
        "run_id": str(seed),
        "episode_return": float(kills),
        "turns": turns,
        "furthest_zone": 1,
        "furthest_floor": 1,
        "max_gold": kills,
        "enemy_kills": kills,
        "item_pickups": 0,
        "item_value": 0,
        "enemy_damage": kills,
        "player_damage": int(status == "dead"),
        "status": status,
    }


def test_masked_random_actions_are_reproducible_and_legal() -> None:
    mask = np.asarray([[0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.int8)
    first = masked_random_actions(mask, np.random.default_rng(9))
    second = masked_random_actions(mask, np.random.default_rng(9))
    assert np.array_equal(first, second)
    assert first[0] in {1, 3}
    assert first[1] == 2


def test_baseline_summary_and_delta_use_gameplay_metrics() -> None:
    reference = summarize_episodes(
        [episode(1, status="dead", turns=5), episode(2, status="dead", turns=7)],
        "masked_random",
    )
    trained = summarize_episodes(
        [episode(1, status="dead", turns=9, kills=1), episode(2, status="won", turns=11)],
        "checkpoint_deterministic",
    )
    delta = compare_summaries(reference, trained)
    assert reference["death_rate"] == 1.0
    assert trained["completion_rate"] == 0.5
    assert trained["mean_turns"] == 10.0
    assert delta["mean_turns_delta"] == 4.0
    assert delta["enemy_kills_delta"] == 1.0


def test_hidden_reset_does_not_mutate_inference_tensor() -> None:
    with torch.inference_mode():
        hidden = torch.ones(3, 2)
    reset = zero_hidden_rows(hidden, [1])
    assert torch.equal(reset, torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]))
    assert torch.equal(hidden, torch.ones(3, 2))


def test_recurrent_state_ablation_uses_a_fresh_state_for_every_action() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.initial_state_calls = 0

        def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            self.initial_state_calls += 1
            return torch.zeros(batch_size, 3, device=device)

    model = FakeModel()
    carried = torch.ones(1, 3)
    assert (
        recurrent_state_for_action(  # type: ignore[arg-type]
            model, carried, "carry", device=torch.device("cpu")
        )
        is carried
    )
    first = recurrent_state_for_action(  # type: ignore[arg-type]
        model, carried, "reset-every-step", device=torch.device("cpu")
    )
    second = recurrent_state_for_action(  # type: ignore[arg-type]
        model, carried * 7, "reset-every-step", device=torch.device("cpu")
    )
    assert torch.equal(first, torch.zeros(1, 3))
    assert torch.equal(second, torch.zeros(1, 3))
    assert first.data_ptr() != second.data_ptr()
    assert model.initial_state_calls == 2


def test_recurrent_state_ablation_rejects_unknown_mode() -> None:
    class FakeModel:
        def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros(batch_size, 3, device=device)

    with pytest.raises(ValueError, match="recurrent-state mode"):
        recurrent_state_for_action(  # type: ignore[arg-type]
            FakeModel(),
            torch.ones(1, 3),
            "forget-sometimes",
            device=torch.device("cpu"),
        )


def test_recurrent_state_can_reset_only_at_floor_boundaries() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.initial_state_calls = 0

        def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            self.initial_state_calls += 1
            return torch.zeros(batch_size, 3, device=device)

    model = FakeModel()
    carried = torch.ones(1, 3)
    within_floor = recurrent_state_after_transition(  # type: ignore[arg-type]
        model,
        carried,
        "reset-on-floor-transition",
        previous_level=(1, 1),
        next_level=(1, 1),
        device=torch.device("cpu"),
    )
    next_floor = recurrent_state_after_transition(  # type: ignore[arg-type]
        model,
        carried,
        "reset-on-floor-transition",
        previous_level=(1, 1),
        next_level=(1, 2),
        device=torch.device("cpu"),
    )
    next_zone = recurrent_state_after_transition(  # type: ignore[arg-type]
        model,
        carried,
        "reset-on-floor-transition",
        previous_level=(1, 4),
        next_level=(2, 1),
        device=torch.device("cpu"),
    )
    assert within_floor is carried
    assert torch.equal(next_floor, torch.zeros(1, 3))
    assert torch.equal(next_zone, torch.zeros(1, 3))
    assert model.initial_state_calls == 2


def test_live_evaluation_uses_partial_final_wave_without_counting_padding() -> None:
    environment = OneStepEnvironment()
    results = evaluate_live_policy(
        environment,  # type: ignore[arg-type]
        seeds=[41_001, 41_002, 41_003],
        max_steps=10,
        policy_seed=7,
        device=torch.device("cpu"),
    )
    assert [result["seed"] for result in results] == [41_001, 41_002, 41_003]
    assert len(environment.reset_seeds) == environment.num_envs


def test_live_evaluation_rejects_unknown_policy_mode() -> None:
    with pytest.raises(ValueError, match="policy_mode"):
        evaluate_live_policy(
            OneStepEnvironment(),  # type: ignore[arg-type]
            seeds=[41_001],
            max_steps=10,
            policy_seed=7,
            device=torch.device("cpu"),
            policy_mode="temperature-seven",
        )


def test_live_evaluation_forwards_qualified_trace_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = object()
    captured: dict[str, object] = {}

    def fake_evaluate(*args, **kwargs):
        del args
        captured.update(kwargs)
        return []

    monkeypatch.setattr(baseline_module, "_evaluate_model_async", fake_evaluate)
    results = evaluate_live_policy(
        OneStepEnvironment(),  # type: ignore[arg-type]
        seeds=[41_001],
        max_steps=10,
        policy_seed=7,
        device=torch.device("cpu"),
        model=object(),  # type: ignore[arg-type]
        trace_prefix=marker,  # type: ignore[arg-type]
    )
    assert results == []
    assert captured["trace_prefix"] is marker


def test_stochastic_policy_samples_are_turn_keyed_and_reproducible() -> None:
    first = [stochastic_policy_sample(91_001, 57_001, turn) for turn in range(8)]
    replay = [stochastic_policy_sample(91_001, 57_001, turn) for turn in range(8)]
    other_policy = [stochastic_policy_sample(91_002, 57_001, turn) for turn in range(8)]
    other_game = [stochastic_policy_sample(91_001, 57_002, turn) for turn in range(8)]

    assert first == replay
    assert first != other_policy
    assert first != other_game
    assert all(0.0 <= value < 1.0 for value in first)


def test_stochastic_policy_sample_rejects_negative_turn() -> None:
    with pytest.raises(ValueError, match="turn"):
        stochastic_policy_sample(91_001, 57_001, -1)


def test_episode_diagnostics_measure_idle_exploration_and_stair_conversion() -> None:
    grid = np.zeros((21, 21, 11), dtype=np.int16)
    player = np.zeros(16, dtype=np.int32)
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 1
    value = {"grid": grid, "player": player}
    accumulator = EpisodeAccumulator(1, "worker-0000", "run-1")
    accumulator.initialize(value, {"zone": 1, "floor": 1})

    accumulator.observe(
        value,
        0.0,
        {"zone": 1, "floor": 1, "raw_events": []},
        int(Action.WAIT),
    )
    stairs = grid.copy()
    stairs[10, 11, GridChannel.TERRAIN_CLASS] = int(Terrain.STAIRS)
    stairs[10, 11, GridChannel.VISIBILITY] = 1
    accumulator.observe(
        {"grid": stairs, "player": player},
        0.0,
        {"zone": 1, "floor": 1, "raw_events": []},
        int(Action.RIGHT),
    )
    player_next = player.copy()
    player_next[PlayerFeature.FLOOR] = 2
    accumulator.observe(
        {"grid": grid, "player": player_next},
        5.0,
        {
            "zone": 1,
            "floor": 2,
            "raw_events": [],
            "extrinsic_reward": 5.0,
            "shaping_reward": 0.0,
        },
        int(Action.DOWN),
    )
    result = accumulator.finish("running")
    assert result["wait_actions"] == 1
    assert result["idle_turns"] == 1
    assert result["staircase_discoveries"] == 1
    assert result["staircase_exits"] == 1
    assert result["stair_discovery_to_exit_turns"] == [1]
    assert result["extrinsic_return"] == 5.0
    assert result["successful_action_sequence"] is None
    assert accumulator.finish("curriculum_complete")["successful_action_sequence"] == [
        int(Action.WAIT),
        int(Action.RIGHT),
        int(Action.DOWN),
    ]


def test_episode_diagnostics_separate_productive_and_repeated_stationary_turns() -> None:
    grid = np.zeros((21, 21, 29), dtype=np.int16)
    player = np.zeros(21, dtype=np.int32)
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 1
    value = {"grid": grid, "player": player}
    accumulator = EpisodeAccumulator(7, "worker-0000", "run-7")
    accumulator.initialize(value, {"zone": 1, "floor": 1})
    accumulator.observe(
        value,
        0.0,
        {"zone": 1, "floor": 1, "raw_events": [{"kind": "enemy_damage", "amount": 1}]},
        int(Action.RIGHT),
    )
    accumulator.observe(value, 0.0, {"zone": 1, "floor": 1, "raw_events": []}, int(Action.LEFT))
    accumulator.observe(value, 0.0, {"zone": 1, "floor": 1, "raw_events": []}, int(Action.LEFT))
    result = accumulator.finish("running")
    assert result["productive_stationary_combat_turns"] == 1
    assert result["unchanged_direction_turns"] == 2
    assert result["repeated_direction_turns"] == 1
    assert result["max_repeated_direction_streak"] == 2


def test_episode_diagnostics_separate_boss_and_add_combat() -> None:
    grid = np.zeros((21, 21, 29), dtype=np.int16)
    player = np.zeros(21, dtype=np.int32)
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 4
    value = {"grid": grid, "player": player}
    accumulator = EpisodeAccumulator(13, "worker-0000", "run-13")
    accumulator.initialize(value, {"zone": 1, "floor": 4})
    accumulator.observe(
        value,
        0.0,
        {
            "zone": 1,
            "floor": 4,
            "raw_events": [
                {"kind": "enemy_damage", "amount": 2, "data": {"boss": True}},
                {"kind": "enemy_damage", "amount": 1, "data": {"boss_add": True}},
                {"kind": "enemy_kill", "amount": 1, "data": {"boss_add": True}},
            ],
        },
        int(Action.RIGHT),
    )
    result = accumulator.finish("running")
    assert result["enemy_damage"] == 3
    assert result["boss_damage"] == 2
    assert result["boss_add_damage"] == 1
    assert result["boss_kills"] == 0
    assert result["boss_add_kills"] == 1

    summary = summarize_episodes([result], "checkpoint")
    assert summary["boss_damage"] == 2
    assert summary["boss_add_damage"] == 1
    assert summary["boss_kills"] == 0
    assert summary["boss_add_kills"] == 1


def _death_metal_observation(*, actor_type: int, health: int) -> dict[str, np.ndarray]:
    grid = np.zeros((21, 21, 29), dtype=np.int16)
    grid[10, 11, GridChannel.ACTOR_CLASS] = int(ActorKind.BOSS)
    grid[10, 11, GridChannel.ACTOR_TYPE] = actor_type
    grid[10, 11, GridChannel.HEALTH] = health
    grid[10, 11, GridChannel.MAX_HEALTH] = 9
    grid[10, 11, GridChannel.VISIBILITY] = 2
    player = np.zeros(21, dtype=np.int32)
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 4
    player[PlayerFeature.TASK] = int(BossType.DEATH_METAL)
    return {"grid": grid, "player": player}


def test_death_metal_phase4_requires_authoritative_damage_and_four_phase_types() -> None:
    accumulator = EpisodeAccumulator(17, "worker-0000", "run-17")
    accumulator.initialize(
        _death_metal_observation(actor_type=101, health=9),
        {"zone": 1, "floor": 4, "boss_type": int(BossType.DEATH_METAL)},
    )
    for actor_type, health, damage in ((102, 6, 3), (103, 4, 2), (104, 2, 2)):
        accumulator.observe(
            _death_metal_observation(actor_type=actor_type, health=health),
            0.0,
            {
                "zone": 1,
                "floor": 4,
                "boss_type": int(BossType.DEATH_METAL),
                "raw_events": [{"kind": "enemy_damage", "amount": damage, "data": {"boss": True}}],
            },
            int(Action.RIGHT),
        )

    result = accumulator.finish("running")
    assert result["boss_actor_types"] == [101, 102, 103, 104]
    assert result["boss_phase_depth"] == 4
    assert result["initial_boss_health"] == 9
    assert result["minimum_boss_health"] == 2
    assert result["boss_damage"] == 7
    assert result["death_metal_phase4_reached"] is True
    summary = summarize_episodes([result], "checkpoint")
    assert summary["death_metal_phase4_rate"] == 1.0
    assert summary["mean_boss_phase_depth"] == 4.0


def test_direct_health_drop_does_not_count_as_death_metal_phase_progression() -> None:
    accumulator = EpisodeAccumulator(18, "worker-0000", "run-18")
    accumulator.initialize(
        _death_metal_observation(actor_type=101, health=9),
        {"zone": 1, "floor": 4, "boss_type": int(BossType.DEATH_METAL)},
    )
    accumulator.observe(
        _death_metal_observation(actor_type=101, health=2),
        0.0,
        {
            "zone": 1,
            "floor": 4,
            "boss_type": int(BossType.DEATH_METAL),
            "raw_events": [{"kind": "enemy_damage", "amount": 7, "data": {"boss": True}}],
        },
        int(Action.RIGHT),
    )

    result = accumulator.finish("running")
    assert result["boss_phase_depth"] == 1
    assert result["death_metal_phase4_reached"] is False
    assert summarize_episodes([result], "checkpoint")["death_metal_phase4_rate"] == 0.0


def test_death_metal_diagnostics_keep_authoritative_offscreen_phase_type() -> None:
    accumulator = EpisodeAccumulator(19, "worker-0000", "run-19")
    accumulator.initialize(
        _death_metal_observation(actor_type=101, health=9),
        {"zone": 1, "floor": 4, "boss_type": int(BossType.DEATH_METAL)},
    )
    for actor_type, health, amount in ((102, 6, 3), (103, 4, 2)):
        accumulator.observe(
            _death_metal_observation(actor_type=actor_type, health=health),
            0.0,
            {
                "zone": 1,
                "floor": 4,
                "boss_type": int(BossType.DEATH_METAL),
                "raw_events": [
                    {
                        "kind": "enemy_damage",
                        "amount": amount,
                        "data": {"boss": True, "actor_type": actor_type},
                    }
                ],
            },
            int(Action.RIGHT),
        )
    observation = _death_metal_observation(actor_type=104, health=2)
    observation["grid"][..., GridChannel.VISIBILITY] = 0
    accumulator.observe(
        observation,
        0.0,
        {
            "zone": 1,
            "floor": 4,
            "boss_type": int(BossType.DEATH_METAL),
            "raw_events": [
                {
                    "kind": "enemy_damage",
                    "amount": 2,
                    "data": {"boss": True, "actor_type": 104},
                }
            ],
        },
        int(Action.RIGHT),
    )

    result = accumulator.finish("running")
    assert result["boss_actor_types"] == [101, 102, 103, 104]
    assert result["death_metal_phase4_reached"] is True


def test_episode_summary_records_action_contract_and_wall_outcomes() -> None:
    grid = np.zeros((21, 21, 29), dtype=np.int16)
    player = np.zeros(21, dtype=np.int32)
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 1
    value = {"grid": grid, "player": player}
    accumulator = EpisodeAccumulator(9, "worker-0000", "run-9")
    accumulator.initialize(value, {"zone": 1, "floor": 1})
    accumulator.observe(
        value,
        0.0,
        {
            "zone": 1,
            "floor": 1,
            "raw_events": [],
            "action_outcome": {"category": "wall_attempt"},
            "action_contract": {
                "newly_learned_invalid_wall": True,
                "masked_direction_count": 1,
                "effective_masked_direction_count": 3,
                "remembered_wall_states": 1,
                "remembered_hazards": 2,
                "navigation_prior_active": True,
                "navigation_masked_directions": [1, 2],
            },
        },
        int(Action.RIGHT),
    )
    summary = summarize_episodes([accumulator.finish("dead")], "checkpoint")
    assert summary["action_outcome_counts"] == {"wall_attempt": 1}
    assert summary["wall_attempt_rate"] == 1
    assert summary["known_invalid_wall_discoveries"] == 1
    assert summary["mean_masked_directions"] == 1
    assert summary["mean_effective_masked_directions"] == 3
    assert summary["navigation_prior_rate"] == 1
    assert summary["mean_navigation_masked_directions"] == 2
    assert summary["mean_max_remembered_wall_states"] == 1
    assert summary["mean_max_remembered_hazards"] == 2


def test_episode_diagnostics_separate_inventory_from_currency_pickups() -> None:
    grid = np.zeros((21, 21, 29), dtype=np.int16)
    player = np.zeros(21, dtype=np.int32)
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 1
    inventory = np.zeros((INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16)
    inventory[0, InventoryFeature.ITEM_TYPE] = 10
    initial = {"grid": grid, "player": player, "inventory": inventory}
    accumulator = EpisodeAccumulator(8, "worker-0000", "run-8")
    accumulator.initialize(initial, {"zone": 1, "floor": 1})

    acquired = inventory.copy()
    acquired[1, InventoryFeature.ITEM_TYPE] = 20
    acquired[2, InventoryFeature.ITEM_TYPE] = 30
    acquired[2, InventoryFeature.QUANTITY] = 2
    accumulator.observe(
        {"grid": grid, "player": player, "inventory": acquired},
        0.0,
        {
            "zone": 1,
            "floor": 1,
            "raw_events": [{"kind": "currency_collected", "amount": 5}],
        },
        int(Action.RIGHT),
    )
    result = accumulator.finish("running")
    assert result["item_pickups"] == 3
    assert result["unique_item_types"] == 2
    assert result["currency_pickups"] == 1
    assert result["currency_value"] == 5


def test_episode_progress_keeps_zone_and_floor_from_the_same_level() -> None:
    grid = np.zeros((21, 21, 29), dtype=np.int16)
    player = np.zeros(21, dtype=np.int32)
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 4
    value = {"grid": grid, "player": player}
    accumulator = EpisodeAccumulator(12, "worker-0000", "run-12")
    accumulator.initialize(value, {"zone": 1, "floor": 4})

    player_next = player.copy()
    player_next[PlayerFeature.ZONE] = 2
    player_next[PlayerFeature.FLOOR] = 1
    accumulator.observe(
        {"grid": grid, "player": player_next},
        10.0,
        {"zone": 2, "floor": 1, "raw_events": []},
        int(Action.DOWN),
    )

    result = accumulator.finish("running")
    summary = summarize_episodes([result], "checkpoint")
    assert (result["furthest_zone"], result["furthest_floor"]) == (2, 1)
    assert summary["mean_progress"] == 5
    assert summary["furthest_floor"] == 5
    assert summary["furthest_level"] == 5
    assert (summary["deepest_zone"], summary["deepest_floor"]) == (2, 1)


def test_episode_summary_marks_repeated_item_transactions() -> None:
    value = episode(1, status="dead", turns=10)
    value["item_pickups"] = 20
    value["unique_item_types"] = 2
    summary = summarize_episodes([value], "checkpoint")
    assert summary["repeat_item_transactions"] == 18
