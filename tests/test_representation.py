from __future__ import annotations

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
    Terrain,
)
from autodancer.training.model import (
    AdapterActorCritic,
    AdapterConfig,
    ModelConfig,
    RecurrentActorCritic,
)
from autodancer.training.ppo import PPOConfig, RecurrentPPO, RolloutBatch
from autodancer.training.representation import NEW_GROUPS, analyze_model


def _adapter() -> AdapterActorCritic:
    return AdapterActorCritic(
        AdapterConfig(
            cell_size=32,
            spatial_size=64,
            hidden_size=32,
            entity_limit=16,
            attention_layers=1,
            attention_heads=4,
            tactical_size=16,
            map_size=16,
            player_size=8,
            inventory_size=8,
        )
    )


def _rollout(model: AdapterActorCritic) -> RolloutBatch:
    time_steps, workers = 1, 2
    grid = torch.zeros(
        time_steps, workers, GRID_SIZE, GRID_SIZE, GRID_CHANNELS, dtype=torch.long
    )
    grid[..., int(GridChannel.TERRAIN_CLASS)] = int(Terrain.FLOOR)
    grid[..., int(GridChannel.VISIBILITY)] = 2
    grid[..., int(GridChannel.FACING) :] = 1
    observations = {
        "grid": grid,
        "map_memory": torch.ones(
            time_steps, workers, MAP_SIZE, MAP_SIZE, MAP_CHANNELS, dtype=torch.long
        ),
        "player": torch.ones(time_steps, workers, PLAYER_FEATURES, dtype=torch.long),
        "inventory": torch.ones(
            time_steps,
            workers,
            INVENTORY_SLOTS,
            INVENTORY_FEATURES,
            dtype=torch.long,
        ),
        "action_mask": torch.ones(time_steps, workers, ACTION_COUNT, dtype=torch.bool),
        "previous_action": torch.zeros(time_steps, workers, dtype=torch.long),
        "previous_reward": torch.zeros(time_steps, workers),
    }
    return RolloutBatch(
        observations=observations,
        actions=torch.zeros(time_steps, workers, dtype=torch.long),
        old_log_probs=torch.zeros(time_steps, workers),
        rewards=torch.ones(time_steps, workers),
        dones=torch.zeros(time_steps, workers),
        terminations=torch.zeros(time_steps, workers, dtype=torch.bool),
        truncation_values=torch.zeros(time_steps, workers),
        episode_starts=torch.ones(time_steps, workers, dtype=torch.bool),
        values=torch.zeros(time_steps, workers),
        hiddens=torch.zeros(time_steps, workers, 2, model.hidden_size),
        next_value=torch.zeros(workers),
    )


def test_representation_diagnostic_separates_unsupported_and_zero_gated_inputs() -> None:
    base = RecurrentActorCritic(
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
    base_report = analyze_model(base, batch_size=2)
    assert all(base_report["groups"][name]["status"] == "unsupported" for name in NEW_GROUPS)

    adapter = _adapter()
    gated_report = analyze_model(adapter, batch_size=2)
    assert all(gated_report["groups"][name]["status"] == "inactive" for name in NEW_GROUPS)
    assert gated_report["adapter_gate_gradient_norm"] > 0


def test_open_adapter_has_sensitivity_and_gradient_reach_for_every_new_group() -> None:
    model = _adapter()
    with torch.no_grad():
        model.adapter_gate.fill_(0.1)
    report = analyze_model(model, batch_size=2)
    assert all(report["groups"][name]["status"] == "material" for name in NEW_GROUPS)


def test_ppo_update_records_representation_gradient_snapshot() -> None:
    model = _adapter()
    with torch.no_grad():
        model.adapter_gate.fill_(0.1)
    algorithm = RecurrentPPO(
        model,
        PPOConfig(
            rollout_length=1,
            sequence_length=1,
            update_epochs=1,
            minibatch_chunks=2,
        ),
        device=torch.device("cpu"),
    )
    metrics = algorithm.update(_rollout(model))
    assert metrics["gradient_adapter_gate"] > 0
    assert all(metrics[f"gradient_{name}"] > 0 for name in NEW_GROUPS)
