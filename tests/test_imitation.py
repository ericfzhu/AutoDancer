from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from autodancer.constants import ACTION_COUNT
from autodancer.observation import observation_space
from autodancer.training.imitation import (
    ImitationConfig,
    RecurrentImitationUpdater,
    imitation_specification,
)
from autodancer.training.imitation_sequences import RecurrentDemonstration
from autodancer.training.model import START_ACTION, ModelConfig, RecurrentActorCritic
from autodancer.training.ppo import PPOConfig, RecurrentPPO


def demonstration() -> RecurrentDemonstration:
    actions = np.asarray([0, 1, 0, 1], dtype=np.int64)
    spaces = observation_space().spaces
    observations = {
        name: np.zeros((len(actions), *spaces[name].shape), dtype=spaces[name].dtype)
        for name in ("grid", "map_memory", "player", "inventory", "action_mask")
    }
    observations["action_mask"][:, :2] = 1
    return RecurrentDemonstration(
        trace_id="a" * 64,
        seed=92043,
        observations=observations,
        actions=actions,
        previous_actions=np.asarray([START_ACTION, 0, 1, 0], dtype=np.int64),
        previous_rewards=np.asarray([0.0, 0.25, -0.1, 0.5], dtype=np.float32),
        episode_starts=np.asarray([True, False, False, False]),
    )


def model() -> RecurrentActorCritic:
    return RecurrentActorCritic(
        ModelConfig(
            cell_size=32,
            spatial_size=64,
            hidden_size=32,
            entity_limit=16,
            attention_layers=1,
            attention_heads=4,
            map_size=0,
        )
    )


def test_imitation_coefficient_decays_to_declared_floor() -> None:
    config = ImitationConfig(coefficient=0.8, final_coefficient=0.2, decay_updates=10)
    assert config.coefficient_at(0) == pytest.approx(0.8)
    assert config.coefficient_at(5) == pytest.approx(0.5)
    assert config.coefficient_at(10) == pytest.approx(0.2)
    assert config.coefficient_at(100) == pytest.approx(0.2)


def test_recurrent_imitation_updates_actor_but_not_critic() -> None:
    torch.manual_seed(17)
    policy = model()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    updater = RecurrentImitationUpdater(
        policy,
        optimizer,
        (demonstration(),),
        ImitationConfig(sequence_length=2, minibatch_sequences=2),
        device=torch.device("cpu"),
    )
    critic_before = {
        name: value.detach().clone()
        for name, value in policy.named_parameters()
        if name.startswith("critic.")
    }
    actor_before = policy.actor[-1].weight.detach().clone()

    metrics = updater.update(0)

    assert metrics["imitation_optimizer_steps"] == 1
    assert metrics["imitation_actor_nll"] > 0
    assert 0 < metrics["imitation_expert_probability"] < 1
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert not torch.equal(policy.actor[-1].weight, actor_before)
    assert all(
        torch.equal(dict(policy.named_parameters())[name], value)
        for name, value in critic_before.items()
    )


def test_imitation_specification_binds_artifact_and_disables_evaluation() -> None:
    item = demonstration()
    config = ImitationConfig()
    specification = imitation_specification(
        {"artifact_sha256": "b" * 64, "manifest_sha256": "c" * 64},
        (item,),
        config,
    )
    assert specification["artifact_sha256"] == "b" * 64
    assert specification["trace_ids"] == [item.trace_id]
    assert specification["transition_count"] == 4
    assert specification["critic_updated"] is False
    assert specification["evaluation_enabled"] is False
    assert ACTION_COUNT == 11


def test_imitation_optimizer_state_resumes_exactly(tmp_path: Path) -> None:
    item = demonstration()
    config = ImitationConfig(sequence_length=2, minibatch_sequences=2)
    specification = imitation_specification(
        {"artifact_sha256": "b" * 64, "manifest_sha256": "c" * 64},
        (item,),
        config,
    )
    algorithm = RecurrentPPO(
        model(),
        PPOConfig(rollout_length=1, sequence_length=1),
        device=torch.device("cpu"),
        checkpoint_metadata={"imitation": specification},
    )
    updater = RecurrentImitationUpdater(
        algorithm.model,
        algorithm.optimizer,
        (item,),
        config,
        device=torch.device("cpu"),
    )
    updater.update(3)
    destination = tmp_path / "imitation.pt"
    algorithm.save(destination)

    restored = RecurrentPPO(
        model(),
        PPOConfig(rollout_length=1, sequence_length=1),
        device=torch.device("cpu"),
        checkpoint_metadata={"imitation": specification},
    )
    restored.load(destination)

    assert all(
        torch.equal(restored.model.state_dict()[name], value)
        for name, value in algorithm.model.state_dict().items()
    )
    assert restored.optimizer.state_dict()["state"].keys() == algorithm.optimizer.state_dict()[
        "state"
    ].keys()


def test_zero_decayed_imitation_does_not_change_policy() -> None:
    policy = model()
    updater = RecurrentImitationUpdater(
        policy,
        torch.optim.Adam(policy.parameters(), lr=1.0e-3),
        (demonstration(),),
        ImitationConfig(final_coefficient=0.0, decay_updates=1),
        device=torch.device("cpu"),
    )
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    metrics = updater.update(1)
    assert metrics["imitation_coefficient"] == 0
    assert metrics["imitation_optimizer_steps"] == 0
    assert all(torch.equal(policy.state_dict()[name], value) for name, value in before.items())
