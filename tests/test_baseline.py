from __future__ import annotations

import numpy as np
import pytest
import torch

from autodancer.constants import (
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    Action,
    GridChannel,
    InventoryFeature,
    PlayerFeature,
    Terrain,
)
from autodancer.training.baseline import (
    EpisodeAccumulator,
    compare_summaries,
    evaluate_live_policy,
    masked_random_actions,
    stochastic_policy_sample,
    summarize_episodes,
    zero_hidden_rows,
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
    accumulator.observe(
        value, 0.0, {"zone": 1, "floor": 1, "raw_events": []}, int(Action.LEFT)
    )
    accumulator.observe(
        value, 0.0, {"zone": 1, "floor": 1, "raw_events": []}, int(Action.LEFT)
    )
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


def test_episode_summary_marks_repeated_item_transactions() -> None:
    value = episode(1, status="dead", turns=10)
    value["item_pickups"] = 20
    value["unique_item_types"] = 2
    summary = summarize_episodes([value], "checkpoint")
    assert summary["repeat_item_transactions"] == 18
