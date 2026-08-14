"""Categorical symbolic encoder and masked actor-critic for Sample Factory."""

from __future__ import annotations

from typing import Any

import torch
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.model.encoder import Encoder
from torch import Tensor, nn


class AutoDancerEncoder(Encoder):
    def __init__(self, cfg: Any, obs_space: Any) -> None:
        super().__init__(cfg)
        del obs_space
        vocabulary_sizes = (4, 11, 32, 4, 2, 3, 3)
        embedding_sizes = (4, 8, 4, 3, 2, 2, 2)
        self.grid_embeddings = nn.ModuleList(
            nn.Embedding(vocabulary, width)
            for vocabulary, width in zip(vocabulary_sizes, embedding_sizes, strict=True)
        )
        embedded_channels = sum(embedding_sizes)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(embedded_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(),
        )
        self.player_encoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.inventory_encoder = nn.Sequential(
            nn.Linear(8 * 3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.fusion = nn.Sequential(nn.Linear(256 + 64 + 64, 256), nn.ReLU())

    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        grid = observations["grid"].long()
        embedded = []
        for channel, embedding in enumerate(self.grid_embeddings):
            values = grid[..., channel].clamp(0, embedding.num_embeddings - 1)
            embedded.append(embedding(values))
        grid_features = torch.cat(embedded, dim=-1).permute(0, 3, 1, 2).contiguous()
        grid_output = self.grid_encoder(grid_features)

        player = observations["player"].float()
        player_scale = torch.tensor(
            [6, 6, 100, 20, 25, 25, 4, 4, 8000, 10, 10, 32, 1, 6, 1, 1],
            device=player.device,
            dtype=player.dtype,
        )
        player_output = self.player_encoder(player / player_scale)
        inventory = observations["inventory"].float().flatten(start_dim=1) / 32.0
        inventory_output = self.inventory_encoder(inventory)
        return self.fusion(torch.cat((grid_output, player_output, inventory_output), dim=1))

    def get_out_size(self) -> int:
        return 256


class MaskedAutoDancerActorCritic(ActorCriticSharedWeights):
    """Apply the observation mask to categorical logits before sampling and PPO."""

    def forward_head(self, normalized_obs_dict: dict[str, Tensor]) -> Tensor:
        self._action_mask = normalized_obs_dict["action_mask"] > 0.5
        return super().forward_head(normalized_obs_dict)

    def forward_tail(
        self, core_output: Tensor, values_only: bool, sample_actions: bool
    ) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()
        result = TensorDict(values=values)
        if values_only:
            return result
        raw_logits = self.action_parameterization.distribution_linear(decoder_output)
        masked_logits = raw_logits.masked_fill(~self._action_mask, -1.0e9)
        self.last_action_distribution = get_action_distribution(
            self.action_space, raw_logits=masked_logits
        )
        result["action_logits"] = masked_logits
        self._maybe_sample_actions(sample_actions, result)
        return result


def make_encoder(cfg: Any, obs_space: Any) -> AutoDancerEncoder:
    return AutoDancerEncoder(cfg, obs_space)


def make_actor_critic(cfg: Any, obs_space: Any, action_space: Any) -> MaskedAutoDancerActorCritic:
    return MaskedAutoDancerActorCritic(global_model_factory(), obs_space, action_space, cfg)
