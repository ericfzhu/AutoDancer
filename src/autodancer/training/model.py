"""Schema-9 recurrent actor-critic with player-visible tactical and audio state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from autodancer.constants import (
    ACTION_COUNT,
    INVENTORY_SLOTS,
    PLAYER_FEATURES,
    TYPE_VOCAB_SIZE,
    ActorKind,
    GridChannel,
    ItemKind,
    MapChannel,
    ObjectKind,
    Terrain,
    TrapKind,
)

ARCHITECTURE_VERSION = 6
A7_ARCHITECTURE_VERSION = 7
A8_ARCHITECTURE_VERSION = 8
START_ACTION = ACTION_COUNT


@dataclass(frozen=True, slots=True)
class ModelConfig:
    cell_size: int = 96
    spatial_size: int = 512
    hidden_size: int = 512
    entity_limit: int = 64
    attention_layers: int = 2
    attention_heads: int = 4
    map_size: int = 128


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """Function-preserving Architecture-7 configuration."""

    cell_size: int = 96
    spatial_size: int = 512
    hidden_size: int = 512
    entity_limit: int = 64
    attention_layers: int = 2
    attention_heads: int = 4
    tactical_size: int = 128
    map_size: int = 128
    player_size: int = 64
    inventory_size: int = 96

    def base_config(self) -> ModelConfig:
        return ModelConfig(
            cell_size=self.cell_size,
            spatial_size=self.spatial_size,
            hidden_size=self.hidden_size,
            entity_limit=self.entity_limit,
            attention_layers=self.attention_layers,
            attention_heads=self.attention_heads,
            map_size=0,
        )


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

    def __init__(self, config: ModelConfig | None = None, *, initialize: bool = True) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.map_size < 0:
            raise ValueError("map_size cannot be negative")
        self.legacy_v2 = self.config.map_size == 0
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
        if not self.legacy_v2:
            self.facing = nn.Embedding(5, 4)
            self.charge_direction = nn.Embedding(5, 4)
            self.shield_direction = nn.Embedding(5, 4)
            self.object_class = nn.Embedding(len(ObjectKind), 6)
            self.object_type = nn.Embedding(TYPE_VOCAB_SIZE, 12)
            self.interaction_flags = nn.Embedding(32, 6)
            self.price_currency = nn.Embedding(TYPE_VOCAB_SIZE, 6)
        cell_features = 68 if self.legacy_v2 else 122
        self.cell_projection = nn.Sequential(
            nn.Linear(cell_features, width), nn.LayerNorm(width), nn.SiLU()
        )
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
        if self.config.map_size:
            self.map_terrain = nn.Embedding(len(Terrain), 4)
            self.map_reveal = nn.Embedding(3, 3)
            self.map_visits = nn.Embedding(16, 6)
            self.map_recency = nn.Embedding(4, 4)
            self.map_player = nn.Embedding(2, 3)
            self.map_encoder = nn.Sequential(
                nn.Conv2d(20, 32, 5, stride=2, padding=2, bias=False),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
                ResidualBlock(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                ResidualBlock(64),
                nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 96),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(96 * 4 * 4, self.config.map_size),
                nn.LayerNorm(self.config.map_size),
                nn.SiLU(),
            )
        self.row_embedding = nn.Embedding(21, width)
        self.column_embedding = nn.Embedding(21, width)
        self.entity_attention = _transformer(
            width, self.config.attention_heads, self.config.attention_layers
        )
        self.player_encoder = nn.Sequential(
            nn.Linear(16 if self.legacy_v2 else PLAYER_FEATURES, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.inventory_class = nn.Embedding(len(ItemKind), 8)
        self.inventory_type = nn.Embedding(TYPE_VOCAB_SIZE, 16)
        self.inventory_slot = nn.Embedding(8 if self.legacy_v2 else INVENTORY_SLOTS, 16)
        self.inventory_projection = nn.Sequential(
            nn.Linear(42 if self.legacy_v2 else 46, width), nn.LayerNorm(width), nn.SiLU()
        )
        self.inventory_attention = _transformer(
            width, self.config.attention_heads, self.config.attention_layers
        )
        self.previous_action = nn.Embedding(ACTION_COUNT + 1, 32)
        self.context_encoder = nn.Sequential(nn.Linear(34, 64), nn.LayerNorm(64), nn.SiLU())
        fused_size = self.config.spatial_size + self.config.map_size + width + 128 + width + 64
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
        if initialize:
            self._initialize()

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def architecture_spec(self) -> dict[str, Any]:
        config = asdict(self.config)
        if not self.config.map_size:
            config.pop("map_size")
            return {"version": 2, "config": config}
        return {"version": ARCHITECTURE_VERSION, "config": config}

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

    def initialize_critic(self) -> None:
        """Initialize only the value head when all policy weights will be transferred."""
        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        features = [
            self.terrain_class(self._bounded(grid, GridChannel.TERRAIN_CLASS, len(Terrain))),
            self.terrain_type(self._bounded(grid, GridChannel.TERRAIN_TYPE, TYPE_VOCAB_SIZE)),
            self.actor_class(self._bounded(grid, GridChannel.ACTOR_CLASS, len(ActorKind))),
            self.actor_type(self._bounded(grid, GridChannel.ACTOR_TYPE, TYPE_VOCAB_SIZE)),
            self.item_class(self._bounded(grid, GridChannel.ITEM_CLASS, len(ItemKind))),
            self.item_type(self._bounded(grid, GridChannel.ITEM_TYPE, TYPE_VOCAB_SIZE)),
            self.trap(self._bounded(grid, GridChannel.TRAP, len(TrapKind))),
            self.visibility(self._bounded(grid, GridChannel.VISIBILITY, 3)),
            self.status(self._bounded(grid, GridChannel.STATUS, 3)),
            numeric,
        ]
        if not self.legacy_v2:
            delay = grid[..., int(GridChannel.BEAT_DELAY)].float()
            interval = grid[..., int(GridChannel.BEAT_INTERVAL)].float()
            tactical_numeric = torch.stack(
                (
                    self._scale(delay),
                    self._scale(interval),
                    delay / interval.clamp_min(1.0),
                    self._scale(grid[..., int(GridChannel.FROZEN_TURNS)]),
                    self._scale(grid[..., int(GridChannel.CONFUSED_TURNS)]),
                    grid[..., int(GridChannel.CHARGE_STATE)].float(),
                ),
                dim=-1,
            )
            features.extend(
                (
                    self.facing(self._bounded(grid, GridChannel.FACING, 5)),
                    tactical_numeric,
                    self.charge_direction(self._bounded(grid, GridChannel.CHARGE_DIRECTION, 5)),
                    self.shield_direction(self._bounded(grid, GridChannel.SHIELD_DIRECTION, 5)),
                    self.object_class(
                        self._bounded(grid, GridChannel.OBJECT_CLASS, len(ObjectKind))
                    ),
                    self.object_type(self._bounded(grid, GridChannel.OBJECT_TYPE, TYPE_VOCAB_SIZE)),
                    self.interaction_flags(self._bounded(grid, GridChannel.INTERACTION_FLAGS, 32)),
                    self.price_currency(
                        self._bounded(grid, GridChannel.PRICE_CURRENCY, TYPE_VOCAB_SIZE)
                    ),
                    torch.stack(
                        (
                            self._scale(grid[..., int(GridChannel.PRICE_AMOUNT)]),
                            self._scale(grid[..., int(GridChannel.PRICE_HEALTH_BP)]),
                            self._scale(grid[..., int(GridChannel.TRAP_ACTIVATION_DS)]),
                            self._scale(grid[..., int(GridChannel.TRAP_FAILURE_DS)]),
                            self._scale(grid[..., int(GridChannel.TELL_ANIMATION_DS)]),
                            grid[..., int(GridChannel.EXPLOSIVE)].float(),
                        ),
                        dim=-1,
                    ),
                )
            )
        return self.cell_projection(torch.cat(features, dim=-1))

    def _entity_features(self, grid: Tensor, cells: Tensor) -> Tensor:
        _, height, width, cell_width = cells.shape
        salient = (
            (grid[..., int(GridChannel.ACTOR_CLASS)] != 0)
            | (grid[..., int(GridChannel.ITEM_CLASS)] != 0)
            | (grid[..., int(GridChannel.TRAP)] != 0)
            | (grid[..., int(GridChannel.TERRAIN_CLASS)] == int(Terrain.STAIRS))
        )
        if not self.legacy_v2:
            salient = salient | (grid[..., int(GridChannel.OBJECT_CLASS)] != 0)
        salient = salient.flatten(1)
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
        if self.legacy_v2:
            inventory = inventory[:, :8, :4]
        classes = inventory[..., 0].long().clamp(0, len(ItemKind) - 1)
        types = inventory[..., 1].long().clamp(0, TYPE_VOCAB_SIZE - 1)
        slots = (
            torch.arange(inventory.shape[1], device=inventory.device)
            .unsqueeze(0)
            .expand(inventory.shape[0], -1)
        )
        numeric = inventory[..., 2:4] if self.legacy_v2 else inventory[..., 2:8]
        tokens = self.inventory_projection(
            torch.cat(
                (
                    self.inventory_class(classes),
                    self.inventory_type(types),
                    self.inventory_slot(slots),
                    self._scale(numeric),
                ),
                dim=-1,
            )
        )
        return self.inventory_attention(tokens).mean(dim=1)

    def _map_features(self, memory: Tensor) -> Tensor:
        if not self.config.map_size:
            raise RuntimeError("Architecture 2 has no persistent map encoder")
        features = torch.cat(
            (
                self.map_terrain(memory[..., int(MapChannel.TERRAIN_CLASS)].long().clamp(0, 3)),
                self.map_reveal(memory[..., int(MapChannel.REVEAL_STATE)].long().clamp(0, 2)),
                self.map_visits(memory[..., int(MapChannel.VISIT_COUNT)].long().clamp(0, 15)),
                self.map_recency(memory[..., int(MapChannel.VISIT_RECENCY)].long().clamp(0, 3)),
                self.map_player(memory[..., int(MapChannel.PLAYER)].long().clamp(0, 1)),
            ),
            dim=-1,
        )
        return self.map_encoder(features.permute(0, 3, 1, 2))

    def encode(self, observation: dict[str, Tensor]) -> Tensor:
        grid = observation["grid"]
        cells = self._cell_features(grid)
        spatial = self.spatial_encoder(cells.permute(0, 3, 1, 2))
        map_memory = self._map_features(observation["map_memory"]) if self.config.map_size else None
        entities = self._entity_features(grid, cells)
        player_values = observation["player"]
        if self.legacy_v2:
            player_values = player_values[..., :16]
        player = self.player_encoder(self._scale(player_values))
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
        features = [spatial]
        if map_memory is not None:
            features.append(map_memory)
        features.extend((entities, player, inventory, context))
        return self.fusion(torch.cat(features, dim=-1))

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


class SensoryResidualAdapter(nn.Module):
    """Encode only schema-9 fields that Architecture 2 cannot observe."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        extra_grid_channels = len(GridChannel) - int(GridChannel.FACING)
        self.tactical = nn.Sequential(
            nn.Conv2d(extra_grid_channels, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, config.tactical_size),
            nn.LayerNorm(config.tactical_size),
            nn.SiLU(),
        )
        self.map = nn.Sequential(
            nn.Conv2d(len(MapChannel), 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, config.map_size),
            nn.LayerNorm(config.map_size),
            nn.SiLU(),
        )
        self.player = nn.Sequential(
            nn.Linear(PLAYER_FEATURES - 16, config.player_size),
            nn.LayerNorm(config.player_size),
            nn.SiLU(),
        )
        # A2 sees four fields in eight slots. The adapter sees the remaining
        # attributes in those slots plus all fields in the five new slots.
        inventory_values = 8 * 4 + (INVENTORY_SLOTS - 8) * 8
        self.inventory = nn.Sequential(
            nn.Linear(inventory_values, config.inventory_size),
            nn.LayerNorm(config.inventory_size),
            nn.SiLU(),
        )
        fused = config.tactical_size + config.map_size + config.player_size + config.inventory_size
        self.output = nn.Sequential(
            nn.Linear(fused, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Tanh(),
        )

    @staticmethod
    def _scale(value: Tensor) -> Tensor:
        value = value.float()
        return torch.sign(value) * torch.log1p(torch.abs(value)) / 10.0

    def forward(self, observation: dict[str, Tensor]) -> Tensor:
        grid = self._scale(observation["grid"][..., int(GridChannel.FACING) :])
        tactical = self.tactical(grid.permute(0, 3, 1, 2))
        memory = self._scale(observation["map_memory"])
        map_features = self.map(memory.permute(0, 3, 1, 2))
        player = self.player(self._scale(observation["player"][..., 16:PLAYER_FEATURES]))
        inventory = observation["inventory"]
        new_inventory = torch.cat(
            (inventory[:, :8, 4:8].flatten(1), inventory[:, 8:, :8].flatten(1)), dim=-1
        )
        inventory_features = self.inventory(self._scale(new_inventory))
        return self.output(torch.cat((tactical, map_features, player, inventory_features), dim=-1))


class AdapterActorCritic(nn.Module):
    """Architecture 2 plus a bounded, initially disabled sensory residual."""

    def __init__(self, config: AdapterConfig | None = None, *, initialize: bool = True) -> None:
        super().__init__()
        self.config = config or AdapterConfig()
        self.base = RecurrentActorCritic(self.config.base_config(), initialize=initialize)
        self.adapter = SensoryResidualAdapter(self.config)
        self.adapter_gate = nn.Parameter(torch.zeros(()))
        if initialize:
            for module in self.adapter.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    nn.init.orthogonal_(module.weight, gain=2**0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def architecture_spec(self) -> dict[str, Any]:
        return {"version": A7_ARCHITECTURE_VERSION, "config": asdict(self.config)}

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> Tensor:
        return self.base.initial_state(batch_size, device=device)

    def encode(self, observation: dict[str, Tensor]) -> Tensor:
        gate = torch.tanh(self.adapter_gate)
        return self.base.encode(observation) + gate * self.adapter(observation)

    def step(self, observation: dict[str, Tensor], state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden, cell = self.base.lstm(self.encode(observation), (state[:, 0], state[:, 1]))
        next_state = torch.stack((hidden, cell), dim=1)
        logits = self.base.actor(hidden).masked_fill(~observation["action_mask"].bool(), -1.0e9)
        return logits, self.base.critic(hidden).squeeze(-1), next_state

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

    def architecture_metrics(self) -> dict[str, float]:
        squared_norm = sum(
            float(parameter.detach().float().square().sum())
            for parameter in self.adapter.parameters()
        )
        return {
            "adapter_gate_raw": float(self.adapter_gate.detach()),
            "adapter_gate": float(torch.tanh(self.adapter_gate.detach())),
            "adapter_parameter_norm": squared_norm**0.5,
        }


class ProjectedAdapterActorCritic(nn.Module):
    """Architecture 2 plus a zero-output, high-dimensional sensory residual."""

    def __init__(self, config: AdapterConfig | None = None, *, initialize: bool = True) -> None:
        super().__init__()
        self.config = config or AdapterConfig()
        self.base = RecurrentActorCritic(self.config.base_config(), initialize=initialize)
        self.adapter = SensoryResidualAdapter(self.config)
        self.adapter_projection = nn.Linear(
            self.config.hidden_size, self.config.hidden_size, bias=False
        )
        if initialize:
            for module in self.adapter.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    nn.init.orthogonal_(module.weight, gain=2**0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            # Exact A2 parity with a full matrix path: the first update can
            # learn many residual directions instead of one scalar gate.
            nn.init.zeros_(self.adapter_projection.weight)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def architecture_spec(self) -> dict[str, Any]:
        return {"version": A8_ARCHITECTURE_VERSION, "config": asdict(self.config)}

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> Tensor:
        return self.base.initial_state(batch_size, device=device)

    def set_base_trainable(self, trainable: bool) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(trainable)

    def encode(self, observation: dict[str, Tensor]) -> Tensor:
        return self.base.encode(observation) + self.adapter_projection(self.adapter(observation))

    def step(self, observation: dict[str, Tensor], state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden, cell = self.base.lstm(self.encode(observation), (state[:, 0], state[:, 1]))
        next_state = torch.stack((hidden, cell), dim=1)
        logits = self.base.actor(hidden).masked_fill(~observation["action_mask"].bool(), -1.0e9)
        return logits, self.base.critic(hidden).squeeze(-1), next_state

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

    def architecture_metrics(self) -> dict[str, float]:
        adapter_squared = sum(
            float(parameter.detach().float().square().sum())
            for parameter in self.adapter.parameters()
        )
        projection = self.adapter_projection.weight.detach().float()
        return {
            "adapter_projection_norm": float(projection.norm()),
            "adapter_projection_max": float(projection.abs().max()),
            "adapter_parameter_norm": adapter_squared**0.5,
        }


PolicyModel = RecurrentActorCritic | AdapterActorCritic | ProjectedAdapterActorCritic


def representation_parameter_groups(model: PolicyModel) -> dict[str, tuple[str, ...]]:
    """Return group-specific parameter prefixes, excluding shared downstream layers."""
    adapted = isinstance(model, (AdapterActorCritic, ProjectedAdapterActorCritic))
    prefix = "base." if adapted else ""
    groups = {
        "local_terrain": tuple(
            prefix + name for name in ("terrain_class.", "terrain_type.", "visibility.")
        ),
        "local_actors": tuple(
            prefix + name for name in ("actor_class.", "actor_type.", "status.")
        ),
        "local_items_traps": tuple(
            prefix + name for name in ("item_class.", "item_type.", "trap.")
        ),
        "base_player": (prefix + "player_encoder.",),
        "base_inventory": tuple(
            prefix + name
            for name in (
                "inventory_class.",
                "inventory_type.",
                "inventory_slot.",
                "inventory_projection.",
                "inventory_attention.",
            )
        ),
        "recurrent_context": tuple(
            prefix + name for name in ("previous_action.", "context_encoder.")
        ),
    }
    if adapted:
        groups.update(
            {
                "tactical_grid": ("adapter.tactical.",),
                "map_memory": ("adapter.map.",),
                "extended_player": ("adapter.player.",),
                "extended_inventory": ("adapter.inventory.",),
                "adapter_gate": ("adapter_gate",),
            }
        )
        if isinstance(model, ProjectedAdapterActorCritic):
            groups.pop("adapter_gate")
            groups["adapter_projection"] = ("adapter_projection.",)
    elif model.architecture_spec()["version"] == ARCHITECTURE_VERSION:
        groups.update(
            {
                "tactical_grid": (
                    "facing.",
                    "charge_direction.",
                    "shield_direction.",
                    "object_class.",
                    "object_type.",
                    "interaction_flags.",
                    "price_currency.",
                ),
                "map_memory": ("map_",),
                # These encoders mix old and new columns, so their gradient
                # norms are shared; counterfactual sensitivity remains isolated.
                "extended_player": ("player_encoder.",),
                "extended_inventory": (
                    "inventory_projection.",
                    "inventory_attention.",
                ),
            }
        )
    return groups


def current_representation_gradient_norms(model: PolicyModel) -> dict[str, float]:
    """Measure group-specific parameter gradients already produced by a loss."""
    result = {}
    parameters = tuple(model.named_parameters())
    for group, prefixes in representation_parameter_groups(model).items():
        squared: Tensor | None = None
        for name, parameter in parameters:
            if parameter.grad is not None and any(
                name == item or name.startswith(item) for item in prefixes
            ):
                term = parameter.grad.detach().double().square().sum()
                squared = term if squared is None else squared + term
        result[group] = 0.0 if squared is None else float(squared.sqrt())
    return result


def model_from_spec(spec: dict[str, Any], *, initialize: bool = True) -> PolicyModel:
    """Construct the exact model described by a checkpoint architecture spec."""
    version = int(spec.get("version", 0))
    config = dict(spec.get("config", {}))
    if version == 2:
        config["map_size"] = 0
        return RecurrentActorCritic(ModelConfig(**config), initialize=initialize)
    if version == ARCHITECTURE_VERSION:
        return RecurrentActorCritic(ModelConfig(**config), initialize=initialize)
    if version == A7_ARCHITECTURE_VERSION:
        return AdapterActorCritic(AdapterConfig(**config), initialize=initialize)
    if version == A8_ARCHITECTURE_VERSION:
        return ProjectedAdapterActorCritic(AdapterConfig(**config), initialize=initialize)
    raise ValueError(f"Unsupported policy architecture version: {version}")
