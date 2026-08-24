"""Checkpoint diagnostics for observation-group influence and gradient reach."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import Tensor

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    MAP_CHANNELS,
    MAP_SIZE,
    PLAYER_FEATURES,
    TYPE_VOCAB_SIZE,
    ActorKind,
    GridChannel,
    ItemKind,
    Terrain,
    TrapKind,
)
from autodancer.training.model import (
    AdapterActorCritic,
    PolicyModel,
    current_representation_gradient_norms,
    model_from_spec,
    representation_parameter_groups,
)

Observation = dict[str, Tensor]
Perturbation = Callable[[Observation, torch.Generator], None]

BASE_GROUPS = (
    "local_terrain",
    "local_actors",
    "local_items_traps",
    "base_player",
    "base_inventory",
    "recurrent_context",
)
NEW_GROUPS = (
    "tactical_grid",
    "map_memory",
    "extended_player",
    "extended_inventory",
)
ALL_GROUPS = BASE_GROUPS + NEW_GROUPS


def _pattern(shape: torch.Size, maximum: int, *, offset: int = 1) -> Tensor:
    values = torch.arange(shape.numel(), dtype=torch.long).reshape(shape)
    return (values + offset).remainder(maximum)


def _base_observation(batch: int, generator: torch.Generator) -> Observation:
    grid = torch.zeros(batch, GRID_SIZE, GRID_SIZE, GRID_CHANNELS, dtype=torch.long)
    grid[..., int(GridChannel.TERRAIN_CLASS)] = int(Terrain.FLOOR)
    grid[..., int(GridChannel.MAX_HEALTH)] = 6
    grid[..., int(GridChannel.VISIBILITY)] = 2
    return {
        "grid": grid,
        "map_memory": torch.zeros(batch, MAP_SIZE, MAP_SIZE, MAP_CHANNELS, dtype=torch.long),
        "player": torch.zeros(batch, PLAYER_FEATURES, dtype=torch.long),
        "inventory": torch.zeros(
            batch, INVENTORY_SLOTS, INVENTORY_FEATURES, dtype=torch.long
        ),
        "action_mask": torch.ones(batch, ACTION_COUNT, dtype=torch.bool),
        "previous_action": torch.randint(0, ACTION_COUNT + 1, (batch,), generator=generator),
        "previous_reward": torch.randn(batch, generator=generator),
    }


def _local_terrain(observation: Observation, _: torch.Generator) -> None:
    grid = observation["grid"]
    grid[..., int(GridChannel.TERRAIN_CLASS)] = _pattern(
        grid[..., int(GridChannel.TERRAIN_CLASS)].shape, len(Terrain)
    )
    grid[..., int(GridChannel.TERRAIN_TYPE)] = _pattern(
        grid[..., int(GridChannel.TERRAIN_TYPE)].shape, TYPE_VOCAB_SIZE, offset=17
    )
    grid[..., int(GridChannel.VISIBILITY)] = _pattern(
        grid[..., int(GridChannel.VISIBILITY)].shape, 3
    )


def _local_actors(observation: Observation, _: torch.Generator) -> None:
    grid = observation["grid"]
    grid[..., int(GridChannel.ACTOR_CLASS)] = _pattern(
        grid[..., int(GridChannel.ACTOR_CLASS)].shape, len(ActorKind)
    )
    grid[..., int(GridChannel.ACTOR_TYPE)] = _pattern(
        grid[..., int(GridChannel.ACTOR_TYPE)].shape, TYPE_VOCAB_SIZE, offset=31
    )
    grid[..., int(GridChannel.HEALTH)] = _pattern(
        grid[..., int(GridChannel.HEALTH)].shape, 8
    )
    grid[..., int(GridChannel.MAX_HEALTH)] = 8
    grid[..., int(GridChannel.STATUS)] = _pattern(
        grid[..., int(GridChannel.STATUS)].shape, 3
    )


def _local_items_traps(observation: Observation, _: torch.Generator) -> None:
    grid = observation["grid"]
    grid[..., int(GridChannel.ITEM_CLASS)] = _pattern(
        grid[..., int(GridChannel.ITEM_CLASS)].shape, len(ItemKind)
    )
    grid[..., int(GridChannel.ITEM_TYPE)] = _pattern(
        grid[..., int(GridChannel.ITEM_TYPE)].shape, TYPE_VOCAB_SIZE, offset=47
    )
    grid[..., int(GridChannel.TRAP)] = _pattern(
        grid[..., int(GridChannel.TRAP)].shape, len(TrapKind)
    )


def _base_player(observation: Observation, _: torch.Generator) -> None:
    observation["player"][:, :16] = _pattern(observation["player"][:, :16].shape, 17)


def _base_inventory(observation: Observation, _: torch.Generator) -> None:
    inventory = observation["inventory"]
    inventory[:, :8, 0] = _pattern(inventory[:, :8, 0].shape, len(ItemKind))
    inventory[:, :8, 1] = _pattern(inventory[:, :8, 1].shape, TYPE_VOCAB_SIZE, offset=61)
    inventory[:, :8, 2:4] = _pattern(inventory[:, :8, 2:4].shape, 9)


def _recurrent_context(observation: Observation, _: torch.Generator) -> None:
    batch = observation["previous_action"].shape[0]
    observation["previous_action"] = torch.arange(batch).remainder(ACTION_COUNT)
    observation["previous_reward"] = torch.linspace(-2.0, 2.0, batch)


def _tactical_grid(observation: Observation, _: torch.Generator) -> None:
    extra = observation["grid"][..., int(GridChannel.FACING) :]
    extra.copy_(_pattern(extra.shape, 8))


def _map_memory(observation: Observation, _: torch.Generator) -> None:
    memory = observation["map_memory"]
    memory.copy_(_pattern(memory.shape, 4))


def _extended_player(observation: Observation, _: torch.Generator) -> None:
    values = observation["player"][:, 16:]
    values.copy_(_pattern(values.shape, 23))


def _extended_inventory(observation: Observation, _: torch.Generator) -> None:
    inventory = observation["inventory"]
    inventory[:, :8, 4:] = _pattern(inventory[:, :8, 4:].shape, 11)
    inventory[:, 8:, 0] = _pattern(inventory[:, 8:, 0].shape, len(ItemKind))
    inventory[:, 8:, 1] = _pattern(
        inventory[:, 8:, 1].shape, TYPE_VOCAB_SIZE, offset=73
    )
    inventory[:, 8:, 2:] = _pattern(inventory[:, 8:, 2:].shape, 11)


PERTURBATIONS: dict[str, Perturbation] = {
    "local_terrain": _local_terrain,
    "local_actors": _local_actors,
    "local_items_traps": _local_items_traps,
    "base_player": _base_player,
    "base_inventory": _base_inventory,
    "recurrent_context": _recurrent_context,
    "tactical_grid": _tactical_grid,
    "map_memory": _map_memory,
    "extended_player": _extended_player,
    "extended_inventory": _extended_inventory,
}


def _gradient_norms(
    model: PolicyModel, observation: Observation, state: Tensor
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    gradient_observation = {key: value.clone() for key, value in observation.items()}
    for index, perturbation in enumerate(PERTURBATIONS.values()):
        perturbation(gradient_observation, torch.Generator().manual_seed(10_000 + index))
    logits, value, next_state = model.step(gradient_observation, state)
    objective = logits.square().mean() + value.square().mean() + next_state.square().mean()
    objective.backward()
    result = current_representation_gradient_norms(model)
    model.zero_grad(set_to_none=True)
    return result


def analyze_model(
    model: PolicyModel,
    *,
    seed: int = 17,
    batch_size: int = 8,
    tolerance: float = 1.0e-8,
    minimum_relative: float = 0.01,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    observation = _base_observation(batch_size, generator)
    state = torch.randn(batch_size, 2, model.hidden_size, generator=generator) * 0.1
    model.eval()
    gradient_norms = _gradient_norms(model, observation, state)
    with torch.inference_mode():
        baseline = model.step(observation, state)
    supported = set(representation_parameter_groups(model))
    groups = {}
    for index, name in enumerate(ALL_GROUPS):
        changed = {key: value.clone() for key, value in observation.items()}
        PERTURBATIONS[name](changed, torch.Generator().manual_seed(seed + index + 1))
        with torch.inference_mode():
            perturbed = model.step(changed, state)
        logit_delta = perturbed[0] - baseline[0]
        value_delta = perturbed[1] - baseline[1]
        state_delta = perturbed[2] - baseline[2]
        sensitivity = max(
            float(logit_delta.abs().max()),
            float(value_delta.abs().max()),
            float(state_delta.abs().max()),
        )
        gradient = gradient_norms.get(name, 0.0)
        is_supported = name in supported
        groups[name] = {
            "supported": is_supported,
            "sensitivity": sensitivity,
            "gradient_norm": gradient,
            "max_logit_delta": float(logit_delta.abs().max()),
            "rms_logit_delta": float(logit_delta.square().mean().sqrt()),
            "max_value_delta": float(value_delta.abs().max()),
            "rms_state_delta": float(state_delta.square().mean().sqrt()),
            "deterministic_action_change_rate": float(
                (baseline[0].argmax(-1) != perturbed[0].argmax(-1)).float().mean()
            ),
        }
    reference_sensitivity = median(groups[name]["sensitivity"] for name in BASE_GROUPS)
    reference_gradient = median(groups[name]["gradient_norm"] for name in BASE_GROUPS)
    for result in groups.values():
        sensitivity = float(result["sensitivity"])
        gradient = float(result["gradient_norm"])
        sensitivity_ratio = sensitivity / max(reference_sensitivity, tolerance)
        gradient_ratio = gradient / max(reference_gradient, tolerance)
        result["relative_sensitivity"] = sensitivity_ratio
        result["relative_gradient"] = gradient_ratio
        result["numerically_active"] = sensitivity > tolerance and gradient > tolerance
        result["material"] = (
            bool(result["supported"])
            and sensitivity_ratio >= minimum_relative
            and gradient_ratio >= minimum_relative
        )
        result["status"] = (
            "unsupported"
            if not result["supported"]
            else "material"
            if result["material"]
            else "trace"
            if result["numerically_active"]
            else "sensitivity_only"
            if sensitivity > tolerance
            else "gradient_only"
            if gradient > tolerance
            else "inactive"
        )
    return {
        "seed": seed,
        "batch_size": batch_size,
        "tolerance": tolerance,
        "minimum_relative": minimum_relative,
        "reference_sensitivity": reference_sensitivity,
        "reference_gradient": reference_gradient,
        "architecture": model.architecture_spec(),
        "groups": groups,
        "adapter_gate_gradient_norm": gradient_norms.get("adapter_gate"),
    }


def analyze_checkpoint(
    path: Path,
    *,
    seed: int = 17,
    batch_size: int = 8,
    tolerance: float = 1.0e-8,
    minimum_relative: float = 0.01,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = model_from_spec(dict(payload["architecture"]), initialize=False)
    model.load_state_dict(payload["model"])
    result = analyze_model(
        model,
        seed=seed,
        batch_size=batch_size,
        tolerance=tolerance,
        minimum_relative=minimum_relative,
    )
    result.update(
        {
            "checkpoint": str(path.resolve()),
            "global_step": int(payload.get("global_step", 0)),
            "updates": int(payload.get("updates", 0)),
            "checkpoint_metadata": payload.get("checkpoint_metadata", {}),
        }
    )
    if isinstance(model, AdapterActorCritic):
        result["architecture_metrics"] = model.architecture_metrics()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure checkpoint sensitivity and gradient reach by observation group"
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--minimum-relative",
        type=float,
        default=0.01,
        help="minimum sensitivity and gradient ratios for material influence (default: 0.01)",
    )
    arguments = parser.parse_args()
    if (
        arguments.batch_size <= 0
        or arguments.tolerance < 0
        or arguments.minimum_relative < 0
    ):
        parser.error("batch size must be positive and thresholds cannot be negative")
    report = {
        "schema_version": 1,
        "method": "controlled_single-group_counterfactual_and_output_gradient",
        "checkpoints": [
            analyze_checkpoint(
                path,
                seed=arguments.seed,
                batch_size=arguments.batch_size,
                tolerance=arguments.tolerance,
                minimum_relative=arguments.minimum_relative,
            )
            for path in arguments.checkpoints
        ],
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(arguments.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
