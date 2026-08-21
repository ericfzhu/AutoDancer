from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from autodancer.constants import ACTION_COUNT, GRID_CHANNELS, GRID_SIZE
from autodancer.training.model import RecurrentActorCritic
from autodancer.training.ppo import (
    PPOConfig,
    RecurrentPPO,
    RolloutBatch,
    generalized_advantage_estimate,
)


def observations(time_steps: int, workers: int) -> dict[str, torch.Tensor]:
    mask = torch.zeros(time_steps, workers, ACTION_COUNT, dtype=torch.int8)
    mask[..., :4] = 1
    return {
        "grid": torch.zeros(
            time_steps, workers, GRID_SIZE, GRID_SIZE, GRID_CHANNELS, dtype=torch.int16
        ),
        "player": torch.zeros(time_steps, workers, 16, dtype=torch.int32),
        "inventory": torch.zeros(time_steps, workers, 8, 3, dtype=torch.int16),
        "action_mask": mask,
    }


def test_masked_policy_never_selects_an_invalid_action() -> None:
    model = RecurrentActorCritic(hidden_size=32, embedding_size=2)
    observation = {key: value[0] for key, value in observations(1, 3).items()}
    observation["action_mask"].zero_()
    observation["action_mask"][:, 7] = 1
    action, _, _, _, _ = model.act(observation, model.initial_state(3))
    assert action.tolist() == [7, 7, 7]


def test_recurrent_gae_stops_at_episode_boundaries() -> None:
    advantages, returns = generalized_advantage_estimate(
        torch.tensor([[1.0], [10.0]]),
        torch.zeros(2, 1),
        torch.tensor([[True], [False]]),
        torch.tensor([2.0]),
        gamma=0.5,
        gae_lambda=1.0,
    )
    assert torch.allclose(advantages, torch.tensor([[1.0], [11.0]]))
    assert torch.equal(returns, advantages)


def test_ppo_updates_parameters_and_checkpoint_resumes_exactly(tmp_path: Path) -> None:
    torch.manual_seed(4)
    config = PPOConfig(
        rollout_length=2,
        sequence_length=1,
        update_epochs=1,
        minibatch_chunks=2,
    )
    model = RecurrentActorCritic(hidden_size=32, embedding_size=2)
    algorithm = RecurrentPPO(model, config, device=torch.device("cpu"))
    batch = RolloutBatch(
        observations=observations(2, 2),
        actions=torch.zeros(2, 2, dtype=torch.long),
        old_log_probs=torch.full((2, 2), -1.3862944),
        rewards=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        dones=torch.tensor([[False, False], [True, True]]),
        episode_starts=torch.tensor([[True, True], [False, False]]),
        values=torch.zeros(2, 2),
        hiddens=torch.zeros(2, 2, 32),
        next_value=torch.zeros(2),
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = algorithm.update(batch)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())

    checkpoint = tmp_path / "checkpoint.pt"
    algorithm.save(checkpoint, metrics={"sentinel": 9})
    expected_random = (random.random(), float(np.random.random()), float(torch.rand(())))
    restored = RecurrentPPO(
        RecurrentActorCritic(hidden_size=32, embedding_size=2),
        config,
        device=torch.device("cpu"),
    )
    assert restored.load(checkpoint) == {"sentinel": 9}
    assert restored.global_step == algorithm.global_step
    assert restored.updates == algorithm.updates
    actual_random = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual_random == expected_random
    for name, value in algorithm.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
