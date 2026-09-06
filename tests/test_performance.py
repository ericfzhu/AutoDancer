from __future__ import annotations

import json

import pytest
import torch

from autodancer.training.model import (
    AdapterActorCritic,
    AdapterConfig,
    ModelConfig,
    ProjectedAdapterActorCritic,
    RecurrentActorCritic,
    evaluate_sequence_batched,
    warm_recurrent_state_batched,
)
from autodancer.training.performance import benchmark_observations, summarize_metrics


@pytest.mark.parametrize("architecture", [2, 6, 7, 8])
@pytest.mark.parametrize("warm", [False, True])
def test_batched_sequence_preserves_outputs_and_gradients(architecture, warm):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    if architecture in (2, 6):
        model = RecurrentActorCritic(
            ModelConfig(
                cell_size=32,
                spatial_size=32,
                hidden_size=32,
                entity_limit=8,
                attention_layers=1,
                map_size=0 if architecture == 2 else 32,
            )
        )
    else:
        cls = AdapterActorCritic if architecture == 7 else ProjectedAdapterActorCritic
        model = cls(
            AdapterConfig(
                cell_size=32,
                spatial_size=32,
                hidden_size=32,
                entity_limit=8,
                attention_layers=1,
                tactical_size=32,
                map_size=32,
                player_size=32,
                inventory_size=32,
            )
        )
        # Exercise learned adapters, not just the zero-output initialization.
        with torch.no_grad():
            if architecture == 7:
                model.adapter_gate.fill_(0.2)
            else:
                model.adapter_projection.weight.normal_(std=0.05)
    model.train()
    observation = benchmark_observations(2, 3, torch.device("cpu"))
    observation["player"][1, 1, 0] = 2
    actions = torch.tensor([[0, 1, 2], [3, 4, 5]])
    starts = torch.tensor([[True, False, True], [False, True, False]])
    state = torch.randn(2, 2, 32)
    stored = torch.randn(2, 3, 2, 32) if warm else None
    args = (observation, actions, state, starts, stored)
    expected = model.evaluate_sequence(*args)
    sum(value.square().mean() for value in expected).backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual = evaluate_sequence_batched(model, *args, encoder_batch_size=4)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=2e-4, atol=2e-5)
    sum(value.square().mean() for value in actual).backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=2e-4, atol=2e-5)
        else:
            assert parameter.grad is None

    model.eval()
    sequence = {key: value[0] for key, value in observation.items()}
    initial = state[:1]
    with torch.inference_mode():
        reference = initial
        for step in range(3):
            _, _, reference = model.step(
                {key: value[step : step + 1] for key, value in sequence.items()}, reference
            )
        actual_state = warm_recurrent_state_batched(model, sequence, initial, encoder_batch_size=2)
    torch.testing.assert_close(actual_state, reference, rtol=2e-4, atol=2e-5)
    empty = {key: value[:0] for key, value in sequence.items()}
    torch.testing.assert_close(warm_recurrent_state_batched(model, empty, initial), initial)


def test_metrics_summary_uses_elapsed_wall_time_and_keeps_missing_stages_unknown(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {
            "global_step": 1024,
            "collector_seconds": 8,
            "collector_steps_per_second": 128,
            "steps_per_second": 102.4,
            "fragment_straggler_seconds": 2,
        },
        {
            "global_step": 2048,
            "collector_seconds": 12,
            "collector_steps_per_second": 1024 / 12,
            "steps_per_second": 2048 / 25,
            "fragment_straggler_seconds": 4,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = summarize_metrics(path)
    assert report["training_loop_seconds"] == 25
    assert report["collection_fraction"] == 0.8
    assert report["ppo_update_seconds"] is None
    assert report["mean_fragment_straggler_seconds"] == 3
