"""Schema-5 recurrent actor-critic for symbolic AutoDancer observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from autodancer.constants import (
    ACTION_COUNT,
    TYPE_VOCAB_SIZE,
    ActorKind,
    GridChannel,
    ItemKind,
    Terrain,
    TrapKind,
)

ARCHITECTURE_VERSION = 2
START_ACTION = ACTION_COUNT


@dataclass(frozen=True, slots=True)
class ModelConfig:
    cell_size: int = 96
    spatial_size: int = 512
    hidden_size: int = 512
    entity_limit: int = 64
    attention_layers: int = 2
    attention_heads: int = 4


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.layers(value))


def _transformer(width: int, heads: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * 4,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, num_layers=layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
    )


class RecurrentActorCritic(nn.Module):
    """Hybrid spatial/entity policy with an LSTM memory state ``[h, c]``."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        width = self.config.cell_size
        self.terrain_class = nn.Embedding(len(Terrain), 4)
        self.terrain_type = nn.Embedding(TYPE_VOCAB_SIZE, 8)
        self.actor_class = nn.Embedding(len(ActorKind), 8)
        self.actor_type = nn.Embedding(TYPE_VOCAB_SIZE, 16)
        self.item_class = nn.Embedding(len(ItemKind), 6)
        self.item_type = nn.Embedding(TYPE_VOCAB_SIZE, 12)
        self.trap = nn.Embedding(len(TrapKind), 4)
        self.visibility = nn.Embedding(3, 3)
        self.status = nn.Embedding(3, 4)
        self.cell_projection = nn.Sequential(nn.Linear(68, width), nn.LayerNorm(width), nn.SiLU())
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(width, 96, 3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            ResidualBlock(96),
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 192, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            ResidualBlock(192),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(192 * 3 * 3, self.config.spatial_size),
            nn.LayerNorm(self.config.spatial_size),
            nn.SiLU(),
        )
        self.row_embedding = nn.Embedding(21, width)
        self.column_embedding = nn.Embedding(21, width)
        self.entity_attention = _transformer(
            width, self.config.attention_heads, self.config.attention_layers
        )
        self.player_encoder = nn.Sequential(
            nn.Linear(16, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU()
        )
        self.inventory_class = nn.Embedding(len(ItemKind), 8)
        self.inventory_type = nn.Embedding(TYPE_VOCAB_SIZE, 16)
        self.inventory_slot = nn.Embedding(8, 16)
        self.inventory_projection = nn.Sequential(
            nn.Linear(42, width), nn.LayerNorm(width), nn.SiLU()
        )
        self.inventory_attention = _transformer(
            width, self.config.attention_heads, self.config.attention_layers
        )
        self.previous_action = nn.Embedding(ACTION_COUNT + 1, 32)
        self.context_encoder = nn.Sequential(nn.Linear(34, 64), nn.LayerNorm(64), nn.SiLU())
        fused_size = self.config.spatial_size + width + 128 + width + 64
        self.fusion = nn.Sequential(
            nn.Linear(fused_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.SiLU(),
        )
        self.lstm = nn.LSTMCell(self.config.hidden_size, self.config.hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256), nn.SiLU(), nn.Linear(256, ACTION_COUNT)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256), nn.SiLU(), nn.Linear(256, 1)
        )
        self._initialize()

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def architecture_spec(self) -> dict[str, Any]:
        return {"version": ARCHITECTURE_VERSION, "config": asdict(self.config)}

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        with torch.no_grad():
            size = self.config.hidden_size
            self.lstm.bias_ih[size : 2 * size].fill_(1.0)

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, 2, self.config.hidden_size, device=device)

    @staticmethod
    def _scale(value: Tensor) -> Tensor:
        value = value.float()
        return torch.sign(value) * torch.log1p(torch.abs(value)) / 10.0

    @staticmethod
    def _bounded(grid: Tensor, channel: GridChannel, maximum: int) -> Tensor:
        return grid[..., int(channel)].long().clamp(0, maximum - 1)

    def _cell_features(self, grid: Tensor) -> Tensor:
        health = grid[..., int(GridChannel.HEALTH)].float()
        maximum = grid[..., int(GridChannel.MAX_HEALTH)].float()
        numeric = torch.stack(
            (self._scale(health), self._scale(maximum), health / maximum.clamp_min(1.0)), dim=-1
        )
        return self.cell_projection(
            torch.cat(
                (
                    self.terrain_class(
                        self._bounded(grid, GridChannel.TERRAIN_CLASS, len(Terrain))
                    ),
                    self.terrain_type(
                        self._bounded(grid, GridChannel.TERRAIN_TYPE, TYPE_VOCAB_SIZE)
                    ),
                    self.actor_class(self._bounded(grid, GridChannel.ACTOR_CLASS, len(ActorKind))),
                    self.actor_type(self._bounded(grid, GridChannel.ACTOR_TYPE, TYPE_VOCAB_SIZE)),
                    self.item_class(self._bounded(grid, GridChannel.ITEM_CLASS, len(ItemKind))),
                    self.item_type(self._bounded(grid, GridChannel.ITEM_TYPE, TYPE_VOCAB_SIZE)),
                    self.trap(self._bounded(grid, GridChannel.TRAP, len(TrapKind))),
                    self.visibility(self._bounded(grid, GridChannel.VISIBILITY, 3)),
                    self.status(self._bounded(grid, GridChannel.STATUS, 3)),
                    numeric,
                ),
                dim=-1,
            )
        )

    def _entity_features(self, grid: Tensor, cells: Tensor) -> Tensor:
        _, height, width, cell_width = cells.shape
        salient = (
            (grid[..., int(GridChannel.ACTOR_CLASS)] != 0)
            | (grid[..., int(GridChannel.ITEM_CLASS)] != 0)
            | (grid[..., int(GridChannel.TRAP)] != 0)
            | (grid[..., int(GridChannel.TERRAIN_CLASS)] == int(Terrain.STAIRS))
        ).flatten(1)
        # The player occupies the centre in live observations. Keeping that token also
        # makes empty synthetic observations well-defined instead of all-padding attention.
        salient = salient.clone()
        salient[:, (height // 2) * width + width // 2] = True
        limit = min(self.config.entity_limit, height * width)
        tie_break = torch.linspace(0.0, -1.0e-4, height * width, device=grid.device)
        indices = (salient.float() + tie_break.unsqueeze(0)).topk(limit, dim=1).indices
        selected = cells.flatten(1, 2).gather(1, indices.unsqueeze(-1).expand(-1, -1, cell_width))
        selected = (
            selected + self.row_embedding(indices // width) + self.column_embedding(indices % width)
        )
        padding = ~salient.gather(1, indices)
        encoded = self.entity_attention(selected, src_key_padding_mask=padding)
        return encoded.masked_fill(padding.unsqueeze(-1), -torch.inf).amax(dim=1).nan_to_num()

    def _inventory_features(self, inventory: Tensor) -> Tensor:
        classes = inventory[..., 0].long().clamp(0, len(ItemKind) - 1)
        types = inventory[..., 1].long().clamp(0, TYPE_VOCAB_SIZE - 1)
        slots = torch.arange(8, device=inventory.device).unsqueeze(0).expand(inventory.shape[0], -1)
        tokens = self.inventory_projection(
            torch.cat(
                (
                    self.inventory_class(classes),
                    self.inventory_type(types),
                    self.inventory_slot(slots),
                    self._scale(inventory[..., 2:4]),
                ),
                dim=-1,
            )
        )
        return self.inventory_attention(tokens).mean(dim=1)

    def encode(self, observation: dict[str, Tensor]) -> Tensor:
        grid = observation["grid"]
        cells = self._cell_features(grid)
        spatial = self.spatial_encoder(cells.permute(0, 3, 1, 2))
        entities = self._entity_features(grid, cells)
        player = self.player_encoder(self._scale(observation["player"]))
        inventory = self._inventory_features(observation["inventory"])
        previous_action = observation.get("previous_action")
        if previous_action is None:
            previous_action = torch.full(
                (grid.shape[0],), START_ACTION, dtype=torch.long, device=grid.device
            )
        previous_reward = observation.get("previous_reward")
        if previous_reward is None:
            previous_reward = torch.zeros(grid.shape[0], device=grid.device)
        context = self.context_encoder(
            torch.cat(
                (
                    self.previous_action(previous_action.long().clamp(0, START_ACTION)),
                    self._scale(previous_reward).unsqueeze(-1),
                    torch.sign(previous_reward).unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        return self.fusion(torch.cat((spatial, entities, player, inventory, context), dim=-1))

    def step(self, observation: dict[str, Tensor], state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden, cell = self.lstm(self.encode(observation), (state[:, 0], state[:, 1]))
        next_state = torch.stack((hidden, cell), dim=1)
        logits = self.actor(hidden).masked_fill(~observation["action_mask"].bool(), -1.0e9)
        return logits, self.critic(hidden).squeeze(-1), next_state

    def forward(self, observation: dict[str, Tensor], state: Tensor) -> tuple[Tensor, Tensor]:
        logits, _, state = self.step(observation, state)
        return logits, state

    def act(
        self, observation: dict[str, Tensor], state: Tensor, *, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        logits, value, next_state = self.step(observation, state)
        distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value, next_state

    def evaluate_sequence(
        self,
        observation: dict[str, Tensor],
        actions: Tensor,
        initial_state: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        state = initial_state
        log_probs, entropies, values = [], [], []
        for step in range(actions.shape[1]):
            state = state * (~episode_starts[:, step]).float().reshape(-1, 1, 1)
            logits, value, state = self.step(
                {key: item[:, step] for key, item in observation.items()}, state
            )
            distribution = Categorical(logits=logits)
            log_probs.append(distribution.log_prob(actions[:, step]))
            entropies.append(distribution.entropy())
            values.append(value)
        return torch.stack(log_probs, 1), torch.stack(entropies, 1), torch.stack(values, 1)
