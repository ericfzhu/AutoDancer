from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from autodancer.training.async_collector import InferenceScheduler
from autodancer.training.model import ModelConfig, RecurrentActorCritic
from autodancer.training.performance import benchmark_observations
from autodancer.training.scaling import memory_requirement


@pytest.mark.parametrize("mode", ["legacy", "packed"])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_scheduler_preserves_outputs_and_retained_results_across_buffer_reuse(mode, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(1)
    torch.manual_seed(81)
    model = (
        RecurrentActorCritic(
            ModelConfig(
                cell_size=16,
                spatial_size=32,
                hidden_size=16,
                entity_limit=8,
                attention_layers=1,
                map_size=0,
            )
        )
        .to(device)
        .eval()
    )
    tensors = {k: v[:, 0] for k, v in benchmark_observations(3, 1, torch.device("cpu")).items()}
    observations = [
        {
            k: v[i].numpy().copy()
            for k, v in tensors.items()
            if k not in {"previous_action", "previous_reward"}
        }
        for i in range(3)
    ]
    hidden = model.initial_state(3, device=device)
    scheduler = InferenceScheduler(
        model,
        device=device,
        max_batch=3,
        batch_delay=0.01,
        deterministic=True,
        transfer_mode=mode,
    )

    def infer(i):
        return scheduler.infer(observations[i], 11, -0.1, hidden[i : i + 1], 0.5)

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(infer, range(3)))
        saved = [(a, lp.clone(), value.clone(), state.clone()) for a, lp, value, state in results]
        for observation in observations:
            observation["player"][0] = 1
            observation["action_mask"][:] = 0
            observation["action_mask"][2] = 1
        with ThreadPoolExecutor(max_workers=3) as executor:
            changed = list(executor.map(infer, range(3)))
        assert all(result[0] == 2 for result in changed)
        for original, retained in zip(results, saved, strict=True):
            assert original[0] == retained[0]
            for actual, expected in zip(original[1:], retained[1:], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reference = dict(tensors)
        reference["previous_action"] = torch.full((3,), 11)
        reference["previous_reward"] = torch.full((3,), -0.1)
        reference = {key: value.to(device) for key, value in reference.items()}
        with torch.inference_mode():
            actions, log_probs, _, values, states = model.act(reference, hidden, deterministic=True)
        for i, result in enumerate(saved):
            assert result[0] == int(actions[i])
            torch.testing.assert_close(result[1].cpu(), log_probs[i].cpu(), rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(result[2].cpu(), values[i].cpu(), rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(result[3], states[i : i + 1], rtol=2e-4, atol=2e-5)
    finally:
        scheduler.close()
    assert scheduler.stats["requests"] == 6
    assert sum(int(size) * count for size, count in scheduler.stats["batch_histogram"].items()) == 6
    assert scheduler.stats["result_transfer_calls"] == (
        18 if mode == "legacy" else scheduler.stats["batches"]
    )
    assert np.isfinite(scheduler.stats["queue_seconds"])


def test_scaling_reserves_memory_beyond_worker_estimate():
    assert memory_requirement(16, 100, 50) == 1650


def test_warmup_verification_rejects_changed_state_and_scheduler_recovers():
    torch.set_num_threads(1)
    model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=32,
            hidden_size=16,
            entity_limit=8,
            attention_layers=1,
            map_size=0,
        )
    ).eval()
    tensors = {
        key: value[0, 0] for key, value in benchmark_observations(1, 1, torch.device("cpu")).items()
    }
    row = {key: value.numpy().copy() for key, value in tensors.items()}
    initial = model.initial_state(1)
    with torch.inference_mode():
        _, _, reference = model.step({key: value[None] for key, value in tensors.items()}, initial)
    scheduler = InferenceScheduler(model, device=torch.device("cpu"), max_batch=1, batch_delay=0)
    try:
        with pytest.raises(AssertionError):
            scheduler.warmup([row], initial, 2, reference + 1, row)
        actual = scheduler.warmup([row], initial, 2, reference, row)
        torch.testing.assert_close(actual, reference)
        assert scheduler.stats["warmup_verified"] == 1
    finally:
        scheduler.close()
