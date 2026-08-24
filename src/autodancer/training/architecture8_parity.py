"""Preflight proof that Architecture 8 preserves A2 and opens its adapter path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from autodancer.training.architecture7_parity import _new_input_perturbation, _observation
from autodancer.training.model import AdapterConfig, ProjectedAdapterActorCritic, model_from_spec
from autodancer.training.ppo import PPOConfig, RecurrentPPO


def verify_checkpoint(path: Path, *, seed: int = 11, tolerance: float = 1.0e-6) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_spec = dict(payload.get("architecture", {}))
    if source_spec.get("version") != 2:
        raise ValueError("A8 parity requires an Architecture-2 source checkpoint")
    source = model_from_spec(source_spec, initialize=False)
    source.load_state_dict(payload["model"])
    source.eval()
    target = ProjectedAdapterActorCritic(
        AdapterConfig(**dict(source_spec["config"])), initialize=True
    )
    algorithm = RecurrentPPO(
        target,
        PPOConfig(rollout_length=1, sequence_length=1),
        device=torch.device("cpu"),
    )
    provenance = algorithm.initialize_for_finetune(path)
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
    projection_initially_zero = bool(
        torch.count_nonzero(target.adapter_projection.weight.detach()) == 0
    )
    names = ("logits", "values", "next_state")
    maximum_error = {
        name: float((left - right).abs().max())
        for name, left, right in zip(names, source_step, target_step, strict=True)
    }
    perturbation_error = {
        name: float((left - right).abs().max())
        for name, left, right in zip(names, target_step, changed_step, strict=True)
    }

    target.zero_grad(set_to_none=True)
    logits, values, next_state = target.step(changed, state)
    (logits.square().mean() + values.square().mean() + next_state.square().mean()).backward()
    projection_gradient = float(target.adapter_projection.weight.grad.norm())
    first_adapter_gradient = sum(
        float(parameter.grad.norm())
        for parameter in target.adapter.parameters()
        if parameter.grad is not None
    )
    with torch.no_grad():
        target.adapter_projection.weight.add_(
            target.adapter_projection.weight.grad, alpha=-1.0e-3
        )
    target.zero_grad(set_to_none=True)
    logits, values, next_state = target.step(changed, state)
    (logits.square().mean() + values.square().mean() + next_state.square().mean()).backward()
    second_adapter_gradient = sum(
        float(parameter.grad.norm())
        for parameter in target.adapter.parameters()
        if parameter.grad is not None
    )
    passed = (
        tensor_parity
        and projection_initially_zero
        and max(maximum_error.values()) <= tolerance
        and max(perturbation_error.values()) <= tolerance
        and float(target_step[0].argmax(-1).ne(source_step[0].argmax(-1)).float().mean()) == 0
        and projection_gradient > 0
        and first_adapter_gradient == 0
        and second_adapter_gradient > 0
    )
    return {
        "schema_version": 1,
        "source": str(path.resolve()),
        "source_architecture": source_spec,
        "target_architecture": target.architecture_spec(),
        "provenance": provenance,
        "tolerance": tolerance,
        "base_tensor_parity": tensor_parity,
        "projection_initially_zero": projection_initially_zero,
        "maximum_error": maximum_error,
        "new_input_perturbation_error": perturbation_error,
        "projection_gradient_first_step": projection_gradient,
        "adapter_gradient_first_step": first_adapter_gradient,
        "adapter_gradient_after_projection_opens": second_adapter_gradient,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify A2-to-A8 projected-adapter parity")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()
    report = verify_checkpoint(arguments.checkpoint, seed=arguments.seed)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
