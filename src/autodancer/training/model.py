"""Masked recurrent actor-critic for symbolic AutoDancer observations."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from autodancer.constants import ACTION_COUNT, GRID_CHANNELS


class RecurrentActorCritic(nn.Module):
    def __init__(self, *, hidden_size: int = 256, embedding_size: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.grid_embeddings = nn.ModuleList(
            [nn.Embedding(32768, embedding_size) for _ in range(GRID_CHANNELS)]
        )
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS * embedding_size, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.player_encoder = nn.Sequential(nn.Linear(16, 64), nn.Tanh())
        self.inventory_encoder = nn.Sequential(nn.Linear(24, 64), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(64 * 4 * 4 + 128, hidden_size), nn.ReLU())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, ACTION_COUNT)
        self.critic = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    @staticmethod
    def _scale_features(value: Tensor) -> Tensor:
        value = value.float()
        return torch.sign(value) * torch.log1p(torch.abs(value)) / 10.0

    def encode(self, grid: Tensor, player: Tensor, inventory: Tensor) -> Tensor:
        channels = []
        for index, embedding in enumerate(self.grid_embeddings):
            values = grid[..., index].long().clamp_(0, embedding.num_embeddings - 1)
            channels.append(embedding(values).permute(0, 3, 1, 2))
        grid_features = self.grid_encoder(torch.cat(channels, dim=1))
        player_features = self.player_encoder(self._scale_features(player))
        inventory_features = self.inventory_encoder(
            self._scale_features(inventory).flatten(start_dim=1)
        )
        return self.fusion(torch.cat((grid_features, player_features, inventory_features), dim=1))

    def step(
        self,
        grid: Tensor,
        player: Tensor,
        inventory: Tensor,
        action_mask: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.gru(self.encode(grid, player, inventory), hidden)
        logits = self.actor(hidden)
        logits = logits.masked_fill(~action_mask.bool(), -1.0e9)
        return logits, self.critic(hidden).squeeze(-1), hidden

    def forward(
        self,
        grid: Tensor,
        player: Tensor,
        inventory: Tensor,
        action_mask: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits, _, hidden = self.step(grid, player, inventory, action_mask, hidden)
        return logits, hidden

    def act(
        self,
        observation: dict[str, Tensor],
        hidden: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        logits, value, next_hidden = self.step(
            observation["grid"],
            observation["player"],
            observation["inventory"],
            observation["action_mask"],
            hidden,
        )
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value, next_hidden

    def evaluate_sequence(
        self,
        observation: dict[str, Tensor],
        actions: Tensor,
        initial_hidden: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = initial_hidden
        log_probs = []
        entropies = []
        values = []
        for step in range(actions.shape[1]):
            hidden = hidden * (~episode_starts[:, step]).float().unsqueeze(-1)
            logits, value, hidden = self.step(
                observation["grid"][:, step],
                observation["player"][:, step],
                observation["inventory"][:, step],
                observation["action_mask"][:, step],
                hidden,
            )
            distribution = Categorical(logits=logits)
            log_probs.append(distribution.log_prob(actions[:, step]))
            entropies.append(distribution.entropy())
            values.append(value)
        return (
            torch.stack(log_probs, dim=1),
            torch.stack(entropies, dim=1),
            torch.stack(values, dim=1),
        )
