"""Recurrent clipped PPO update and checkpoint state."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from autodancer.training.model import PolicyModel, current_representation_gradient_norms


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")


@dataclass(slots=True)
class RolloutBatch:
    observations: dict[str, Tensor]
    actions: Tensor
    old_log_probs: Tensor
    rewards: Tensor
    dones: Tensor
    terminations: Tensor
    truncation_values: Tensor
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
    terminations: Tensor | None = None,
    truncation_values: Tensor | None = None,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    if terminations is None:
        # Backward-compatible interpretation for isolated callers: every
        # episode boundary is a true MDP termination.
        terminations = dones
    if truncation_values is None:
        truncation_values = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)
    future_value = next_value
    for step in reversed(range(rewards.shape[0])):
        boundary = dones[step].bool()
        terminated = terminations[step].bool()
        bootstrap_value = torch.where(
            terminated,
            torch.zeros_like(future_value),
            torch.where(boundary, truncation_values[step], future_value),
        )
        delta = rewards[step] + gamma * bootstrap_value - values[step]
        trace_continuing = 1.0 - boundary.float()
        last_advantage = delta + gamma * gae_lambda * trace_continuing * last_advantage
        advantages[step] = last_advantage
        future_value = values[step]
    return advantages, advantages + values


class RecurrentPPO:
    def __init__(
        self,
        model: PolicyModel,
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
            terminations=rollout.terminations,
            truncation_values=rollout.truncation_values,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        raw_advantages = advantages
        advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std() + 1.0e-8
        )
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
            "gradient_norm_preclip": [],
        }
        representation_gradients: dict[str, float] | None = None
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
                if representation_gradients is None:
                    representation_gradients = current_representation_gradient_norms(self.model)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.max_grad_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    metrics["policy_loss"].append(float(policy_loss))
                    metrics["value_loss"].append(float(value_loss))
                    metrics["entropy"].append(float(entropy_mean))
                    metrics["approx_kl"].append(float(((ratio - 1) - log_ratio).mean()))
                    metrics["clip_fraction"].append(
                        float((torch.abs(ratio - 1) > config.clip_range).float().mean())
                    )
                    metrics["gradient_norm_preclip"].append(float(gradient_norm))
        self.global_step += time_steps * workers
        self.updates += 1
        result = {name: float(np.mean(values)) for name, values in metrics.items()}
        with torch.no_grad():
            target_variance = returns.var(unbiased=False)
            explained_variance = torch.where(
                target_variance > 1.0e-12,
                1.0 - (returns - rollout.values).var(unbiased=False) / target_variance,
                torch.zeros_like(target_variance),
            )
            diagnostics = {
                "explained_variance": explained_variance,
                "value_mean": rollout.values.mean(),
                "value_std": rollout.values.std(unbiased=False),
                "value_max_abs": rollout.values.abs().max(),
                "return_mean": returns.mean(),
                "return_std": returns.std(unbiased=False),
                "return_max_abs": returns.abs().max(),
                "advantage_raw_mean": raw_advantages.mean(),
                "advantage_raw_std": raw_advantages.std(unbiased=False),
                "advantage_raw_max_abs": raw_advantages.abs().max(),
                "reward_mean": rollout.rewards.mean(),
                "reward_std": rollout.rewards.std(unbiased=False),
                "reward_max_abs": rollout.rewards.abs().max(),
            }
        result.update({name: float(value) for name, value in diagnostics.items()})
        result.update(
            {
                f"gradient_{name}": value
                for name, value in (representation_gradients or {}).items()
            }
        )
        return result

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
                "Checkpoint model architecture is incompatible with the schema-9 policy"
            )
        saved_metadata = dict(payload.get("checkpoint_metadata", {}))
        if any(saved_metadata.get(key) != value for key, value in self.checkpoint_metadata.items()):
            raise ValueError("Checkpoint training metadata does not match the current run")
        if payload.get("config") != asdict(self.config):
            raise ValueError("Checkpoint PPO configuration does not match the current run")
        self.model.load_state_dict(payload["model"])
        self.checkpoint_metadata = saved_metadata
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload["global_step"])
        self.updates = int(payload["updates"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        # Checkpoints loaded directly onto CUDA also move ByteTensor RNG states.
        # PyTorch's RNG restoration APIs require CPU byte tensors even when the
        # model and optimizer live on the GPU.
        torch.set_rng_state(payload["torch_rng"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
        return dict(payload.get("metrics", {}))

    def initialize_from(self, path: str | Path) -> dict[str, Any]:
        """Load representation and actor weights without old reward-value state."""
        path = Path(path)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        source_spec = payload.get("architecture")
        target_spec = self.model.architecture_spec()
        v2_to_v8 = (
            isinstance(source_spec, dict)
            and source_spec.get("version") == 2
            and target_spec.get("version") == 8
            and source_spec.get("config")
            == {
                key: target_spec["config"][key]
                for key in (
                    "cell_size",
                    "spatial_size",
                    "hidden_size",
                    "entity_limit",
                    "attention_layers",
                    "attention_heads",
                )
            }
        )
        if v2_to_v8:
            source = dict(payload["model"])
            target = self.model.state_dict()
            transferred = {
                f"base.{name}": value
                for name, value in source.items()
                if not name.startswith("critic.")
            }
            missing, unexpected = self.model.load_state_dict(transferred, strict=False)
            allowed_missing = {
                name
                for name in target
                if name.startswith(
                    ("base.critic.", "adapter.", "adapter_projection.")
                )
            }
            if unexpected or set(missing) != allowed_missing:
                raise ValueError(
                    "Architecture-2 checkpoint did not populate the A8 actor with a fresh critic"
                )
            provenance = {
                "path": str(path.resolve()),
                "sha256": _checkpoint_sha256(path),
                "global_step": int(payload.get("global_step", 0)),
                "updates": int(payload.get("updates", 0)),
                "reward": payload.get("checkpoint_metadata", {}).get("reward"),
                "architecture_upgrade": "v2_to_v8_actor_parity_fresh_critic",
            }
            self.checkpoint_metadata["initialization"] = provenance
            return provenance
        v2_to_v7 = (
            isinstance(source_spec, dict)
            and source_spec.get("version") == 2
            and target_spec.get("version") == 7
            and source_spec.get("config")
            == {
                key: target_spec["config"][key]
                for key in (
                    "cell_size",
                    "spatial_size",
                    "hidden_size",
                    "entity_limit",
                    "attention_layers",
                    "attention_heads",
                )
            }
        )
        if v2_to_v7:
            source = dict(payload["model"])
            target = self.model.state_dict()
            transferred = {f"base.{name}": value for name, value in source.items()}
            missing, unexpected = self.model.load_state_dict(transferred, strict=False)
            allowed_missing = {
                name
                for name in target
                if name == "adapter_gate" or name.startswith("adapter.")
            }
            if unexpected or set(missing) != allowed_missing:
                raise ValueError("Architecture-2 checkpoint did not exactly populate A7 base")
            provenance = {
                "path": str(path.resolve()),
                "sha256": _checkpoint_sha256(path),
                "global_step": int(payload.get("global_step", 0)),
                "updates": int(payload.get("updates", 0)),
                "reward": payload.get("checkpoint_metadata", {}).get("reward"),
                "architecture_upgrade": "v2_to_v7_zero_gated_exact",
            }
            self.checkpoint_metadata["initialization"] = provenance
            return provenance
        exact_architecture = source_spec == target_spec
        v2_upgrade = (
            isinstance(source_spec, dict)
            and source_spec.get("version") == 2
            and target_spec.get("version") == 6
            and source_spec.get("config")
            == {key: value for key, value in target_spec["config"].items() if key != "map_size"}
        )
        v4_upgrade = (
            isinstance(source_spec, dict)
            and source_spec.get("version") == 4
            and target_spec.get("version") == 6
            and source_spec.get("config") == target_spec.get("config")
        )
        v5_upgrade = (
            isinstance(source_spec, dict)
            and source_spec.get("version") == 5
            and target_spec.get("version") == 6
            and source_spec.get("config") == target_spec.get("config")
        )
        sensory_upgrade = v2_upgrade or v4_upgrade or v5_upgrade
        if not exact_architecture and not sensory_upgrade:
            raise ValueError(
                "Initialization checkpoint architecture is incompatible with the schema-9 policy"
            )
        source = dict(payload["model"])
        target = self.model.state_dict()
        transferred = {
            name: value
            for name, value in source.items()
            if not name.startswith("critic.")
            and name in target
            and target[name].shape == value.shape
        }
        if sensory_upgrade:
            # Retain every compatible portion of the source representation while
            # leaving newly added observations at their Architecture-6 initialization.
            for name in (
                "cell_projection.0.weight",
                "player_encoder.0.weight",
                "inventory_slot.weight",
                "inventory_projection.0.weight",
            ):
                source_value = source.get(name)
                if source_value is None:
                    continue
                expanded = target[name].clone()
                slices = tuple(slice(0, size) for size in source_value.shape)
                expanded[slices] = source_value
                transferred[name] = expanded
        missing, unexpected = self.model.load_state_dict(transferred, strict=False)
        allowed_missing = {
            name
            for name in target
            if name.startswith(
                (
                    "critic.",
                    "map_",
                    "facing.",
                    "charge_direction.",
                    "shield_direction.",
                    "object_class.",
                    "object_type.",
                    "interaction_flags.",
                    "price_currency.",
                )
            )
            or name == "fusion.0.weight"
        }
        if (
            unexpected
            or (
                exact_architecture
                and set(missing) != {name for name in target if name.startswith("critic.")}
            )
            or (sensory_upgrade and not set(missing).issubset(allowed_missing))
        ):
            raise ValueError("Initialization checkpoint did not match policy parameters")
        provenance = {
            "path": str(path.resolve()),
            "sha256": _checkpoint_sha256(path),
            "global_step": int(payload.get("global_step", 0)),
            "updates": int(payload.get("updates", 0)),
            "reward": payload.get("checkpoint_metadata", {}).get("reward"),
            "architecture_upgrade": (
                "v2_to_v6_player_parity"
                if v2_upgrade
                else "v4_to_v6_interactions_audio"
                if v4_upgrade
                else "v5_to_v6_audio"
                if v5_upgrade
                else None
            ),
        }
        self.checkpoint_metadata["initialization"] = provenance
        return provenance

    def initialize_for_finetune(self, path: str | Path) -> dict[str, Any]:
        """Preserve the complete source function while resetting optimizer state."""
        path = Path(path)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        source_spec = payload.get("architecture")
        target_spec = self.model.architecture_spec()
        exact = source_spec == target_spec
        a2_to_a8 = (
            isinstance(source_spec, dict)
            and source_spec.get("version") == 2
            and target_spec.get("version") == 8
            and source_spec.get("config")
            == {
                key: target_spec["config"][key]
                for key in (
                    "cell_size",
                    "spatial_size",
                    "hidden_size",
                    "entity_limit",
                    "attention_layers",
                    "attention_heads",
                )
            }
        )
        if exact:
            self.model.load_state_dict(payload["model"])
            upgrade = "exact_function_preserving_finetune"
        elif a2_to_a8:
            target = self.model.state_dict()
            transferred = {f"base.{name}": value for name, value in payload["model"].items()}
            missing, unexpected = self.model.load_state_dict(transferred, strict=False)
            allowed_missing = {
                name
                for name in target
                if name.startswith(("adapter.", "adapter_projection."))
            }
            if unexpected or set(missing) != allowed_missing:
                raise ValueError("Architecture-2 checkpoint did not exactly populate A8 base")
            upgrade = "v2_to_v8_zero_projection_exact"
        else:
            raise ValueError("Fine-tune checkpoint architecture is incompatible with target")
        provenance = {
            "path": str(path.resolve()),
            "sha256": _checkpoint_sha256(path),
            "global_step": int(payload.get("global_step", 0)),
            "updates": int(payload.get("updates", 0)),
            "reward": payload.get("checkpoint_metadata", {}).get("reward"),
            "architecture_upgrade": upgrade,
        }
        self.checkpoint_metadata["initialization"] = provenance
        return provenance
