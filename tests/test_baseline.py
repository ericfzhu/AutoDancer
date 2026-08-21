from __future__ import annotations

import numpy as np
import torch

from autodancer.training.baseline import (
    compare_summaries,
    masked_random_actions,
    summarize_episodes,
    zero_hidden_rows,
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
