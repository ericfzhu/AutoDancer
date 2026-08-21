"""Recurrent clipped PPO update and checkpoint state."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from autodancer.training.model import RecurrentActorCritic


@dataclass(frozen=True, slots=True)
class PPOConfig:
    rollout_length: int = 128
    sequence_length: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3.0e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_chunks: int = 8

    def __post_init__(self) -> None:
        if self.rollout_length <= 0 or self.sequence_length <= 0:
            raise ValueError("rollout and sequence lengths must be positive")
        if self.rollout_length % self.sequence_length:
            raise ValueError("rollout_length must be divisible by sequence_length")


@dataclass(slots=True)
class RolloutBatch:
    observations: dict[str, Tensor]
    actions: Tensor
    old_log_probs: Tensor
    rewards: Tensor
    dones: Tensor
    episode_starts: Tensor
    values: Tensor
    hiddens: Tensor
    next_value: Tensor


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)
    future_value = next_value
    for step in reversed(range(rewards.shape[0])):
        continuing = 1.0 - dones[step].float()
        delta = rewards[step] + gamma * future_value * continuing - values[step]
        last_advantage = delta + gamma * gae_lambda * continuing * last_advantage
        advantages[step] = last_advantage
        future_value = values[step]
    return advantages, advantages + values


class RecurrentPPO:
    def __init__(
        self,
        model: RecurrentActorCritic,
        config: PPOConfig,
        *,
        device: torch.device,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.global_step = 0
        self.updates = 0

    def update(self, rollout: RolloutBatch) -> dict[str, float]:
        config = self.config
        advantages, returns = generalized_advantage_estimate(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            rollout.next_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        time_steps, workers = rollout.actions.shape
        chunks = [
            (worker, start)
            for worker in range(workers)
            for start in range(0, time_steps, config.sequence_length)
        ]
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        self.model.train()
        for _ in range(config.update_epochs):
            random.shuffle(chunks)
            for batch_start in range(0, len(chunks), config.minibatch_chunks):
                selected = chunks[batch_start : batch_start + config.minibatch_chunks]
                observation = {
                    key: torch.stack(
                        [
                            value[
                                start : start + config.sequence_length,
                                worker,
                            ]
                            for worker, start in selected
                        ]
                    ).to(self.device)
                    for key, value in rollout.observations.items()
                }
                actions = self._chunks(rollout.actions, selected).long().to(self.device)
                old_log_probs = self._chunks(rollout.old_log_probs, selected).to(self.device)
                batch_advantages = self._chunks(advantages, selected).to(self.device)
                batch_returns = self._chunks(returns, selected).to(self.device)
                episode_starts = (
                    self._chunks(rollout.episode_starts, selected).bool().to(self.device)
                )
                initial_hidden = torch.stack(
                    [rollout.hiddens[start, worker] for worker, start in selected]
                ).to(self.device)
                log_probs, entropy, values = self.model.evaluate_sequence(
                    observation, actions, initial_hidden, episode_starts
                )
                log_ratio = log_probs - old_log_probs
                ratio = log_ratio.exp()
                unclipped = ratio * batch_advantages
                clipped = ratio.clamp(1 - config.clip_range, 1 + config.clip_range)
                policy_loss = -torch.minimum(unclipped, clipped * batch_advantages).mean()
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    metrics["policy_loss"].append(float(policy_loss))
                    metrics["value_loss"].append(float(value_loss))
                    metrics["entropy"].append(float(entropy_mean))
                    metrics["approx_kl"].append(float(((ratio - 1) - log_ratio).mean()))
                    metrics["clip_fraction"].append(
                        float((torch.abs(ratio - 1) > config.clip_range).float().mean())
                    )
        self.global_step += time_steps * workers
        self.updates += 1
        return {name: float(np.mean(values)) for name, values in metrics.items()}

    def _chunks(self, value: Tensor, chunks: list[tuple[int, int]]) -> Tensor:
        length = self.config.sequence_length
        return torch.stack([value[start : start + length, worker] for worker, start in chunks])

    def save(self, path: str | Path, *, metrics: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = {
            "model": self.model.state_dict(),
            "architecture": self.model.architecture_spec(),
            "checkpoint_metadata": self.checkpoint_metadata,
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "global_step": self.global_step,
            "updates": self.updates,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "metrics": metrics or {},
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("architecture") != self.model.architecture_spec():
            raise ValueError(
                "Checkpoint model architecture is incompatible with the schema-5 policy"
            )
        if payload.get("checkpoint_metadata", {}) != self.checkpoint_metadata:
            raise ValueError("Checkpoint training metadata does not match the current run")
        if payload.get("config") != asdict(self.config):
            raise ValueError("Checkpoint PPO configuration does not match the current run")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload["global_step"])
        self.updates = int(payload["updates"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return dict(payload.get("metrics", {}))
