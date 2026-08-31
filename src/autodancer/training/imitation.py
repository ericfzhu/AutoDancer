"""Actor-only recurrent imitation from qualified live-game demonstrations."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from autodancer.training.imitation_sequences import RecurrentDemonstration
from autodancer.training.model import PolicyModel


@dataclass(frozen=True, slots=True)
class ImitationConfig:
    """A decaying auxiliary objective applied after an on-policy PPO update."""

    coefficient: float = 1.0
    final_coefficient: float = 0.0
    decay_updates: int = 60
    epochs: int = 1
    sequence_length: int = 32
    minibatch_sequences: int = 8
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.coefficient <= 0.0:
            raise ValueError("imitation coefficient must be positive")
        if not 0.0 <= self.final_coefficient <= self.coefficient:
            raise ValueError("final imitation coefficient must be in [0, coefficient]")
        if self.decay_updates <= 0:
            raise ValueError("imitation decay updates must be positive")
        if self.epochs <= 0 or self.sequence_length <= 0 or self.minibatch_sequences <= 0:
            raise ValueError("imitation epochs, sequence length, and batch size must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("imitation max gradient norm must be positive")

    def coefficient_at(self, update_index: int) -> float:
        if update_index < 0:
            raise ValueError("imitation update index cannot be negative")
        progress = min(update_index / self.decay_updates, 1.0)
        return self.coefficient + progress * (self.final_coefficient - self.coefficient)


def imitation_specification(
    manifest: dict[str, Any],
    demonstrations: tuple[RecurrentDemonstration, ...],
    config: ImitationConfig,
) -> dict[str, Any]:
    """Return the immutable checkpoint/lineage identity for an imitation objective."""

    return {
        "schema_version": 1,
        "objective": "qualified-live-recurrent-actor-imitation-v1",
        "artifact_sha256": str(manifest["artifact_sha256"]),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "trace_ids": [item.trace_id for item in demonstrations],
        "seeds": [item.seed for item in demonstrations],
        "transition_count": sum(item.length for item in demonstrations),
        "config": asdict(config),
        "critic_updated": False,
        "evaluation_enabled": False,
    }


class RecurrentImitationUpdater:
    """Apply behavior cloning without placing demonstrations in PPO rollouts."""

    def __init__(
        self,
        model: PolicyModel,
        optimizer: torch.optim.Optimizer,
        demonstrations: tuple[RecurrentDemonstration, ...],
        config: ImitationConfig,
        *,
        device: torch.device,
    ) -> None:
        if not demonstrations:
            raise ValueError("recurrent imitation requires at least one demonstration")
        for demonstration in demonstrations:
            demonstration.validate()
        self.model = model
        self.optimizer = optimizer
        self.demonstrations = demonstrations
        self.config = config
        self.device = device
        self._observations = tuple(self._tensor_observations(item) for item in demonstrations)
        self._actions = tuple(
            torch.from_numpy(np.asarray(item.actions)).long() for item in demonstrations
        )
        self._starts = tuple(
            torch.from_numpy(np.asarray(item.episode_starts)).bool() for item in demonstrations
        )

    @staticmethod
    def _tensor_observations(demonstration: RecurrentDemonstration) -> dict[str, Tensor]:
        result = {
            name: torch.from_numpy(np.asarray(value))
            for name, value in demonstration.observations.items()
        }
        result["previous_action"] = torch.from_numpy(
            np.asarray(demonstration.previous_actions)
        ).long()
        result["previous_reward"] = torch.from_numpy(
            np.asarray(demonstration.previous_rewards)
        ).float()
        return result

    def _chunk_initial_states(self, index: int) -> dict[int, Tensor]:
        observation = self._observations[index]
        starts = self._starts[index]
        length = self.demonstrations[index].length
        state = self.model.initial_state(1, device=self.device)
        states: dict[int, Tensor] = {}
        with torch.no_grad():
            for step in range(length):
                if step % self.config.sequence_length == 0:
                    states[step] = state.detach().clone()
                if bool(starts[step]):
                    state.zero_()
                _, _, state = self.model.step(
                    {
                        name: value[step : step + 1].to(self.device)
                        for name, value in observation.items()
                    },
                    state,
                )
        return states

    def update(self, update_index: int) -> dict[str, float]:
        coefficient = self.config.coefficient_at(update_index)
        base_metrics = {
            "imitation_coefficient": coefficient,
            "imitation_trace_count": float(len(self.demonstrations)),
            "imitation_transition_count": float(
                sum(item.length for item in self.demonstrations)
            ),
        }
        if coefficient <= 0.0:
            return {
                **base_metrics,
                "imitation_actor_nll": 0.0,
                "imitation_expert_probability": 0.0,
                "imitation_entropy": 0.0,
                "imitation_gradient_norm_preclip": 0.0,
                "imitation_optimizer_steps": 0.0,
            }

        chunks = [
            (index, start)
            for index, demonstration in enumerate(self.demonstrations)
            for start in range(0, demonstration.length, self.config.sequence_length)
        ]
        nll_values: list[float] = []
        probability_values: list[float] = []
        entropy_values: list[float] = []
        gradient_values: list[float] = []
        optimizer_steps = 0
        self.model.train()
        for _ in range(self.config.epochs):
            initial_states = {
                index: self._chunk_initial_states(index)
                for index in range(len(self.demonstrations))
            }
            random.shuffle(chunks)
            for batch_start in range(0, len(chunks), self.config.minibatch_sequences):
                selected = chunks[
                    batch_start : batch_start + self.config.minibatch_sequences
                ]
                sequence_losses: list[Tensor] = []
                sequence_probabilities: list[Tensor] = []
                sequence_entropies: list[Tensor] = []
                for index, start in selected:
                    end = min(
                        start + self.config.sequence_length,
                        self.demonstrations[index].length,
                    )
                    observation = {
                        name: value[start:end].unsqueeze(0).to(self.device)
                        for name, value in self._observations[index].items()
                    }
                    actions = self._actions[index][start:end].unsqueeze(0).to(self.device)
                    starts = self._starts[index][start:end].unsqueeze(0).to(self.device)
                    log_prob, entropy, _ = self.model.evaluate_sequence(
                        observation,
                        actions,
                        initial_states[index][start],
                        starts,
                    )
                    sequence_losses.append(-log_prob.mean())
                    sequence_probabilities.append(log_prob.exp().mean())
                    sequence_entropies.append(entropy.mean())
                actor_nll = torch.stack(sequence_losses).mean()
                loss = coefficient * actor_nll
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                optimizer_steps += 1
                nll_values.append(float(actor_nll.detach()))
                probability_values.append(
                    float(torch.stack(sequence_probabilities).mean().detach())
                )
                entropy_values.append(float(torch.stack(sequence_entropies).mean().detach()))
                gradient_values.append(float(gradient_norm))
        return {
            **base_metrics,
            "imitation_actor_nll": float(np.mean(nll_values)),
            "imitation_expert_probability": float(np.mean(probability_values)),
            "imitation_entropy": float(np.mean(entropy_values)),
            "imitation_gradient_norm_preclip": float(np.mean(gradient_values)),
            "imitation_optimizer_steps": float(optimizer_steps),
        }
