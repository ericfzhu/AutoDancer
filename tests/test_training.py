from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
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
    GridChannel,
    PlayerFeature,
)
from autodancer.training.model import START_ACTION, ModelConfig, RecurrentActorCritic
from autodancer.training.ppo import (
    PPOConfig,
    RecurrentPPO,
    RolloutBatch,
    generalized_advantage_estimate,
)
from autodancer.training.train import RolloutCollector


def observations(time_steps: int, workers: int) -> dict[str, torch.Tensor]:
    mask = torch.zeros(time_steps, workers, ACTION_COUNT, dtype=torch.int8)
    mask[..., :4] = 1
    return {
        "grid": torch.zeros(
            time_steps, workers, GRID_SIZE, GRID_SIZE, GRID_CHANNELS, dtype=torch.int16
        ),
        "map_memory": torch.zeros(
            time_steps, workers, MAP_SIZE, MAP_SIZE, MAP_CHANNELS, dtype=torch.int16
        ),
        "player": torch.zeros(time_steps, workers, PLAYER_FEATURES, dtype=torch.int32),
        "inventory": torch.zeros(
            time_steps, workers, INVENTORY_SLOTS, INVENTORY_FEATURES, dtype=torch.int16
        ),
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


def test_policy_encoding_uses_persistent_map_memory() -> None:
    torch.manual_seed(8)
    model = small_model().eval()
    base = {key: value[0] for key, value in observations(1, 1).items()}
    changed = {key: value.clone() for key, value in base.items()}
    changed["map_memory"][0, 32, 40, 0] = 1
    changed["map_memory"][0, 32, 40, 1] = 1
    with torch.inference_mode():
        assert not torch.allclose(model.encode(base), model.encode(changed))


def test_policy_encoding_uses_visible_tactical_and_hud_state() -> None:
    torch.manual_seed(9)
    model = small_model().eval()
    base = {key: value[0] for key, value in observations(1, 1).items()}
    changed = {key: value.clone() for key, value in base.items()}
    changed["grid"][0, 10, 11, int(GridChannel.FACING)] = 2
    changed["grid"][0, 10, 11, int(GridChannel.BEAT_INTERVAL)] = 3
    changed["grid"][0, 10, 11, int(GridChannel.OBJECT_CLASS)] = 2
    changed["grid"][0, 10, 11, int(GridChannel.INTERACTION_FLAGS)] = 5
    changed["grid"][0, 10, 11, int(GridChannel.PRICE_AMOUNT)] = 50
    changed["inventory"][0, 8, 0] = 1
    changed["inventory"][0, 8, 6] = 1
    changed["player"][0, 18] = 300
    changed["player"][0, PlayerFeature.SHOP_MUSIC_VOLUME_BP] = 5000
    with torch.inference_mode():
        assert not torch.allclose(model.encode(base), model.encode(changed))


def test_legacy_checkpoint_is_rejected(tmp_path: Path) -> None:
    config = PPOConfig(rollout_length=1, sequence_length=1)
    algorithm = RecurrentPPO(small_model(), config, device=torch.device("cpu"))
    path = tmp_path / "legacy.pt"
    torch.save({"model": algorithm.model.state_dict(), "config": {}}, path)
    with pytest.raises(ValueError, match="architecture is incompatible"):
        algorithm.load(path)


def test_checkpoint_rejects_a_different_reward_profile(tmp_path: Path) -> None:
    config = PPOConfig(rollout_length=1, sequence_length=1)
    path = tmp_path / "checkpoint.pt"
    RecurrentPPO(
        small_model(),
        config,
        device=torch.device("cpu"),
        checkpoint_metadata={"reward": {"version": 1}},
    ).save(path)
    incompatible = RecurrentPPO(
        small_model(),
        config,
        device=torch.device("cpu"),
        checkpoint_metadata={"reward": {"version": 2}},
    )
    with pytest.raises(ValueError, match="training metadata"):
        incompatible.load(path)


def test_checkpoint_warm_start_transfers_policy_but_resets_critic(tmp_path: Path) -> None:
    config = PPOConfig(rollout_length=1, sequence_length=1)
    source = RecurrentPPO(small_model(), config, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.fill_(0.25)
    path = tmp_path / "source.pt"
    source.save(path)

    target = RecurrentPPO(small_model(), config, device=torch.device("cpu"))
    critic_before = {
        name: value.clone()
        for name, value in target.model.state_dict().items()
        if name.startswith("critic.")
    }
    provenance = target.initialize_from(path)
    for name, value in target.model.state_dict().items():
        if name.startswith("critic."):
            assert torch.equal(value, critic_before[name])
        else:
            assert torch.equal(value, source.model.state_dict()[name])
    assert target.global_step == 0
    assert target.updates == 0
    assert not target.optimizer.state
    assert provenance["global_step"] == 0


def test_architecture_two_checkpoint_can_partially_initialize_map_policy(
    tmp_path: Path,
) -> None:
    source = RecurrentActorCritic(
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
    source_state = source.state_dict()
    source_spec = source.architecture_spec()
    with torch.no_grad():
        source.actor[-1].weight.fill_(0.125)
    source_state["actor.2.weight"] = source.actor[-1].weight.clone()
    path = tmp_path / "architecture-2.pt"
    torch.save(
        {
            "model": source_state,
            "architecture": source_spec,
            "global_step": 250_000,
            "updates": 244,
            "checkpoint_metadata": {"reward": {"version": 2}},
        },
        path,
    )

    target = small_model()
    map_before = target.map_terrain.weight.detach().clone()
    facing_before = target.facing.weight.detach().clone()
    algorithm = RecurrentPPO(
        target,
        PPOConfig(rollout_length=1, sequence_length=1),
        device=torch.device("cpu"),
    )
    provenance = algorithm.initialize_from(path)
    assert torch.equal(target.actor[-1].weight, source.actor[-1].weight)
    assert torch.equal(target.map_terrain.weight, map_before)
    assert torch.equal(target.facing.weight, facing_before)
    assert torch.equal(target.cell_projection[0].weight[:, :68], source.cell_projection[0].weight)
    assert provenance["architecture_upgrade"] == "v2_to_v6_player_parity"


def test_architecture_four_checkpoint_can_initialize_interaction_policy(
    tmp_path: Path,
) -> None:
    source = small_model()
    source_state = {
        name: value.clone()
        for name, value in source.state_dict().items()
        if not name.startswith(
            ("object_class.", "object_type.", "interaction_flags.", "price_currency.")
        )
    }
    source_state["cell_projection.0.weight"] = source_state["cell_projection.0.weight"][:, :86]
    source_state["player_encoder.0.weight"] = source_state["player_encoder.0.weight"][:, :20]
    source_spec = source.architecture_spec()
    source_spec["version"] = 4
    path = tmp_path / "architecture-4.pt"
    torch.save(
        {
            "model": source_state,
            "architecture": source_spec,
            "global_step": 1,
            "updates": 1,
        },
        path,
    )

    target = small_model()
    object_before = target.object_class.weight.detach().clone()
    algorithm = RecurrentPPO(
        target,
        PPOConfig(rollout_length=1, sequence_length=1),
        device=torch.device("cpu"),
    )
    provenance = algorithm.initialize_from(path)
    assert torch.equal(target.object_class.weight, object_before)
    assert torch.equal(
        target.cell_projection[0].weight[:, :86], source_state["cell_projection.0.weight"]
    )
    assert provenance["architecture_upgrade"] == "v4_to_v6_interactions_audio"


def test_architecture_five_checkpoint_can_initialize_audio_policy(tmp_path: Path) -> None:
    source = small_model()
    source_state = {name: value.clone() for name, value in source.state_dict().items()}
    source_state["player_encoder.0.weight"] = source_state["player_encoder.0.weight"][:, :20]
    source_spec = source.architecture_spec()
    source_spec["version"] = 5
    path = tmp_path / "architecture-5.pt"
    torch.save({"model": source_state, "architecture": source_spec}, path)

    target = small_model()
    audio_before = target.player_encoder[0].weight[:, 20].detach().clone()
    algorithm = RecurrentPPO(
        target,
        PPOConfig(rollout_length=1, sequence_length=1),
        device=torch.device("cpu"),
    )
    provenance = algorithm.initialize_from(path)
    assert torch.equal(
        target.player_encoder[0].weight[:, :20], source_state["player_encoder.0.weight"]
    )
    assert torch.equal(target.player_encoder[0].weight[:, 20], audio_before)
    assert provenance["architecture_upgrade"] == "v5_to_v6_audio"


def test_warm_started_checkpoint_resumes_with_provenance_and_rejects_other_arm(
    tmp_path: Path,
) -> None:
    config = PPOConfig(rollout_length=1, sequence_length=1)
    source_path = tmp_path / "source.pt"
    RecurrentPPO(small_model(), config, device=torch.device("cpu")).save(source_path)
    arm_a = {"reward": {"version": 4, "weights": {"stair_potential_max": 0.5}}}
    trained = RecurrentPPO(
        small_model(), config, device=torch.device("cpu"), checkpoint_metadata=arm_a
    )
    trained.initialize_from(source_path)
    checkpoint = tmp_path / "warm.pt"
    trained.save(checkpoint)

    resumed = RecurrentPPO(
        small_model(), config, device=torch.device("cpu"), checkpoint_metadata=arm_a
    )
    resumed.load(checkpoint)
    assert "initialization" in resumed.checkpoint_metadata

    arm_b = {"reward": {"version": 4, "weights": {"stair_potential_max": 1.0}}}
    incompatible = RecurrentPPO(
        small_model(), config, device=torch.device("cpu"), checkpoint_metadata=arm_b
    )
    with pytest.raises(ValueError, match="training metadata"):
        incompatible.load(checkpoint)


def test_rollout_collects_reward_components_without_overwriting_values() -> None:
    class FakeModel:
        def eval(self) -> None:
            pass

        def initial_state(self, batch_size: int, *, device=None) -> torch.Tensor:
            return torch.zeros(batch_size, 2, 1, device=device)

        def act(self, observation, state):
            del observation
            return (
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1),
                torch.zeros(1),
                torch.tensor([0.5]),
                state,
            )

        def step(self, observation, state):
            del observation
            return torch.zeros(1, ACTION_COUNT), torch.tensor([0.5]), state

    class FakeEnvironment:
        num_envs = 1
        worker_ids = ["worker-0000"]

        def reset(self, seeds):
            del seeds
            policy_context = {"previous_action", "previous_reward"}
            return (
                {
                    key: value[0].numpy()
                    for key, value in observations(1, 1).items()
                    if key not in policy_context
                },
                [{}],
            )

        def step(self, actions):
            del actions
            value, _ = self.reset([1])
            return (
                value,
                np.asarray([0.02], dtype=np.float32),
                np.asarray([False]),
                np.asarray([False]),
                [{"reward_components": {"enemy_damage": 0.025}}],
            )

    collector = RolloutCollector(
        FakeEnvironment(),  # type: ignore[arg-type]
        FakeModel(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        seed=1,
    )
    rollout = collector.collect(1)
    assert rollout.values.tolist() == [[0.5]]
    assert collector.last_reward_components == {"enemy_damage": 0.025}


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
