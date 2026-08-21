from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from autodancer.constants import ACTION_COUNT, GRID_CHANNELS, GRID_SIZE, INVENTORY_FEATURES
from autodancer.training.model import START_ACTION, ModelConfig, RecurrentActorCritic
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
        "inventory": torch.zeros(time_steps, workers, 8, INVENTORY_FEATURES, dtype=torch.int16),
        "action_mask": mask,
        "previous_action": torch.full((time_steps, workers), START_ACTION),
        "previous_reward": torch.zeros(time_steps, workers),
    }


def small_model() -> RecurrentActorCritic:
    return RecurrentActorCritic(
        ModelConfig(
            cell_size=32,
            spatial_size=64,
            hidden_size=32,
            entity_limit=16,
            attention_layers=1,
            attention_heads=4,
        )
    )


def test_masked_policy_never_selects_an_invalid_action() -> None:
    model = small_model()
    observation = {key: value[0] for key, value in observations(1, 3).items()}
    observation["action_mask"].zero_()
    observation["action_mask"][:, 7] = 1
    action, _, _, _, _ = model.act(observation, model.initial_state(3))
    assert action.tolist() == [7, 7, 7]


def test_policy_uses_exact_types_and_recurrent_context() -> None:
    torch.manual_seed(7)
    model = small_model().eval()
    base = {key: value[0] for key, value in observations(1, 1).items()}
    changed = {key: value.clone() for key, value in base.items()}
    changed["grid"][0, 10, 11, 3] = 1234
    changed["grid"][0, 10, 11, 2] = 2
    changed["previous_action"][0] = 3
    changed["previous_reward"][0] = 1.0
    with torch.inference_mode():
        base_encoding = model.encode(base)
        changed_encoding = model.encode(changed)
    assert not torch.allclose(base_encoding, changed_encoding)
    assert model.initial_state(1).shape == (1, 2, 32)


def test_legacy_checkpoint_is_rejected(tmp_path: Path) -> None:
    config = PPOConfig(rollout_length=1, sequence_length=1)
    algorithm = RecurrentPPO(small_model(), config, device=torch.device("cpu"))
    path = tmp_path / "legacy.pt"
    torch.save({"model": algorithm.model.state_dict(), "config": {}}, path)
    with pytest.raises(ValueError, match="architecture is incompatible"):
        algorithm.load(path)


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
    model = small_model()
    algorithm = RecurrentPPO(model, config, device=torch.device("cpu"))
    batch = RolloutBatch(
        observations=observations(2, 2),
        actions=torch.zeros(2, 2, dtype=torch.long),
        old_log_probs=torch.full((2, 2), -1.3862944),
        rewards=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        dones=torch.tensor([[False, False], [True, True]]),
        episode_starts=torch.tensor([[True, True], [False, False]]),
        values=torch.zeros(2, 2),
        hiddens=torch.zeros(2, 2, 2, 32),
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
        small_model(),
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
