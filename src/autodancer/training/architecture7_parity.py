"""Preflight proof that a zero-gated A7 exactly preserves an A2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
)
from autodancer.training.model import AdapterActorCritic, AdapterConfig, model_from_spec
from autodancer.training.ppo import PPOConfig, RecurrentPPO


def _observation(batch: int, *, generator: torch.Generator) -> dict[str, torch.Tensor]:
    grid = torch.zeros(batch, GRID_SIZE, GRID_SIZE, GRID_CHANNELS, dtype=torch.long)
    grid[..., int(GridChannel.HEALTH)] = torch.randint(
        0, 8, grid[..., int(GridChannel.HEALTH)].shape, generator=generator
    )
    grid[..., int(GridChannel.MAX_HEALTH)] = 8
    player = torch.zeros(batch, PLAYER_FEATURES, dtype=torch.long)
    inventory = torch.zeros(batch, INVENTORY_SLOTS, INVENTORY_FEATURES, dtype=torch.long)
    return {
        "grid": grid,
        "map_memory": torch.zeros(batch, MAP_SIZE, MAP_SIZE, MAP_CHANNELS, dtype=torch.long),
        "player": player,
        "inventory": inventory,
        "action_mask": torch.ones(batch, ACTION_COUNT, dtype=torch.bool),
        "previous_action": torch.randint(0, ACTION_COUNT + 1, (batch,), generator=generator),
        "previous_reward": torch.randn(batch, generator=generator),
    }


def _new_input_perturbation(
    observation: dict[str, torch.Tensor], *, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in observation.items()}
    changed["grid"][..., int(GridChannel.FACING) :] = torch.randint(
        0,
        8,
        changed["grid"][..., int(GridChannel.FACING) :].shape,
        generator=generator,
    )
    changed["map_memory"] = torch.randint(
        0, 8, changed["map_memory"].shape, generator=generator
    )
    changed["player"][:, 16:] = torch.randint(
        0, 32, changed["player"][:, 16:].shape, generator=generator
    )
    changed["inventory"][:, :8, 4:] = torch.randint(
        0, 32, changed["inventory"][:, :8, 4:].shape, generator=generator
    )
    changed["inventory"][:, 8:, :] = torch.randint(
        0, 32, changed["inventory"][:, 8:, :].shape, generator=generator
    )
    return changed


def verify_checkpoint(path: Path, *, seed: int = 7, tolerance: float = 1.0e-6) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_spec = dict(payload.get("architecture", {}))
    if source_spec.get("version") != 2:
        raise ValueError("A7 parity requires an Architecture-2 source checkpoint")
    source = model_from_spec(source_spec, initialize=False)
    source.load_state_dict(payload["model"])
    source.eval()
    base_config = dict(source_spec["config"])
    target = AdapterActorCritic(AdapterConfig(**base_config), initialize=True)
    algorithm = RecurrentPPO(
        target,
        PPOConfig(rollout_length=1, sequence_length=1),
        device=torch.device("cpu"),
    )
    provenance = algorithm.initialize_from(path)
    target.eval()

    tensor_parity = all(
        torch.equal(value, target.state_dict()[f"base.{name}"])
        for name, value in source.state_dict().items()
    )
    generator = torch.Generator().manual_seed(seed)
    observation = _observation(4, generator=generator)
    changed = _new_input_perturbation(observation, generator=generator)
    state = torch.randn(4, 2, source.hidden_size, generator=generator)
    with torch.inference_mode():
        source_step = source.step(observation, state)
        target_step = target.step(observation, state)
        changed_step = target.step(changed, state)

    names = ("logits", "values", "next_state")
    maximum_error = {
        name: float((left - right).abs().max())
        for name, left, right in zip(names, source_step, target_step, strict=True)
    }
    perturbation_error = {
        name: float((left - right).abs().max())
        for name, left, right in zip(names, target_step, changed_step, strict=True)
    }
    deterministic_actions_equal = torch.equal(source_step[0].argmax(-1), target_step[0].argmax(-1))

    # Sequence parity includes recurrent resets, not merely independent states.
    sequence_observation = {
        key: torch.stack((value, changed[key], value), dim=1)
        for key, value in observation.items()
    }
    actions = torch.zeros(4, 3, dtype=torch.long)
    episode_starts = torch.tensor(
        [[True, False, True], [False, False, False], [True, True, False], [False, True, False]]
    )
    with torch.inference_mode():
        source_sequence = source.evaluate_sequence(
            sequence_observation, actions, state, episode_starts
        )
        target_sequence = target.evaluate_sequence(
            sequence_observation, actions, state, episode_starts
        )
    sequence_maximum_error = max(
        float((left - right).abs().max())
        for left, right in zip(source_sequence, target_sequence, strict=True)
    )
    passed = (
        tensor_parity
        and deterministic_actions_equal
        and max(maximum_error.values()) <= tolerance
        and max(perturbation_error.values()) <= tolerance
        and sequence_maximum_error <= tolerance
        and float(target.adapter_gate.detach()) == 0.0
    )
    return {
        "schema_version": 1,
        "source": str(path.resolve()),
        "source_architecture": source_spec,
        "target_architecture": target.architecture_spec(),
        "provenance": provenance,
        "tolerance": tolerance,
        "base_tensor_parity": tensor_parity,
        "maximum_error": maximum_error,
        "new_input_perturbation_error": perturbation_error,
        "sequence_maximum_error": sequence_maximum_error,
        "deterministic_actions_equal": deterministic_actions_equal,
        "adapter_gate": float(target.adapter_gate.detach()),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify A2-to-A7 zero-gate parity")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    report = verify_checkpoint(arguments.checkpoint, seed=arguments.seed)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
