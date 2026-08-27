"""Reproducible live-game baseline evaluation for Bard policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from autodancer.constants import (
    ACTION_COUNT,
    Action,
    ActorKind,
    BossType,
    GridChannel,
    InventoryFeature,
    PlayerFeature,
    Terrain,
)
from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.bridge import CURRICULUM_PROFILES
from autodancer.live.native_pipe import NativePipeError
from autodancer.live.protocol import SUPPORTED_GAME_VERSION, SUPPORTED_STEAM_BUILD, ProtocolError
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.progress import deeper_level, level_progress
from autodancer.rewards import load_reward_config
from autodancer.training.action_contract import (
    ACTION_CONTRACTS,
    ActionContractMemory,
)
from autodancer.training.async_collector import InferenceScheduler
from autodancer.training.dashboard import DashboardServer, DashboardState
from autodancer.training.model import START_ACTION, PolicyModel, model_from_spec
from autodancer.training.natural_prefix import (
    NATURAL_PREFIX_RECURRENT_MODES,
    DeathMetalPhaseTracker,
    NaturalPrefixConfig,
    NaturalPrefixError,
)
from autodancer.training.train import default_mod_dir, replace_observation_rows, resolve_device

RECURRENT_STATE_MODES = ("carry", "reset-on-floor-transition", "reset-every-step")


@dataclass(slots=True)
class EpisodeAccumulator:
    seed: int
    worker_id: str
    run_id: str
    episode_return: float = 0.0
    turns: int = 0
    furthest_zone: int = 0
    furthest_floor: int = 0
    boss_type: int = 0
    max_gold: int = 0
    enemy_kills: int = 0
    item_pickups: int = 0
    currency_pickups: int = 0
    currency_value: int = 0
    enemy_damage: int = 0
    boss_damage: int = 0
    boss_add_damage: int = 0
    boss_kills: int = 0
    boss_add_kills: int = 0
    player_damage: int = 0
    extrinsic_return: float = 0.0
    shaping_return: float = 0.0
    natural_prefix: dict[str, Any] = field(default_factory=dict)
    boss_actor_types: set[int] = field(default_factory=set)
    initial_boss_health: int | None = None
    minimum_boss_health: int | None = None
    action_counts: list[int] = field(default_factory=lambda: [0] * ACTION_COUNT)
    unchanged_position_turns: int = 0
    max_unchanged_position_streak: int = 0
    idle_turns: int = 0
    productive_stationary_combat_turns: int = 0
    productive_stationary_interaction_turns: int = 0
    unchanged_direction_turns: int = 0
    repeated_direction_turns: int = 0
    special_no_effect_turns: int = 0
    action_outcome_counts: dict[str, int] = field(default_factory=dict)
    known_invalid_wall_discoveries: int = 0
    masked_direction_observations: int = 0
    effective_masked_direction_observations: int = 0
    navigation_prior_turns: int = 0
    navigation_masked_direction_observations: int = 0
    max_remembered_wall_states: int = 0
    max_remembered_hazards: int = 0
    max_repeated_direction_streak: int = 0
    staircase_discoveries: int = 0
    staircase_exits: int = 0
    trapdoor_descents: int = 0
    unknown_descents: int = 0
    stair_discovery_to_exit_turns: list[int] = field(default_factory=list)
    _visited_positions: set[tuple[int, int, int, int]] = field(default_factory=set)
    _last_position: tuple[int, int, int, int] | None = None
    _unchanged_position_streak: int = 0
    _last_unchanged_direction: int | None = None
    _repeated_direction_streak: int = 0
    _floor: tuple[int, int] = (0, 0)
    _stairs_seen_on_floor: bool = False
    _pending_stair_turn: int | None = None
    _inventory_quantities: dict[int, int] = field(default_factory=dict)
    _seen_item_types: set[int] = field(default_factory=set)
    _collected_item_types: set[int] = field(default_factory=set)

    @staticmethod
    def _position(
        observation: dict[str, np.ndarray], info: dict[str, Any]
    ) -> tuple[int, int, int, int]:
        player = observation["player"]
        return (
            int(info.get("zone") or player[PlayerFeature.ZONE]),
            int(info.get("floor") or player[PlayerFeature.FLOOR]),
            int(player[PlayerFeature.X]),
            int(player[PlayerFeature.Y]),
        )

    @staticmethod
    def _has_visible_stairs(observation: dict[str, np.ndarray]) -> bool:
        grid = observation.get("grid")
        if grid is None:
            return False
        return bool(
            np.any(
                (grid[..., GridChannel.TERRAIN_CLASS] == int(Terrain.STAIRS))
                & (grid[..., GridChannel.VISIBILITY] > 0)
            )
        )

    @staticmethod
    def _inventory_counts(observation: dict[str, np.ndarray]) -> dict[int, int]:
        counts: dict[int, int] = {}
        inventory = observation.get("inventory")
        if inventory is None:
            return counts
        for item in inventory:
            item_type = int(item[InventoryFeature.ITEM_TYPE])
            if item_type == 0:
                continue
            quantity = max(int(item[InventoryFeature.QUANTITY]), 1)
            counts[item_type] = counts.get(item_type, 0) + quantity
        return counts

    def _observe_boss_state(self, observation: dict[str, np.ndarray]) -> None:
        grid = observation.get("grid")
        if grid is None:
            return
        mask = (
            (grid[..., GridChannel.ACTOR_CLASS] == int(ActorKind.BOSS))
            & (grid[..., GridChannel.HEALTH] > 0)
            & (grid[..., GridChannel.VISIBILITY] == 2)
        )
        cells = np.argwhere(mask)
        if not len(cells):
            return
        health = min(int(grid[row, column, GridChannel.HEALTH]) for row, column in cells)
        self.initial_boss_health = (
            health if self.initial_boss_health is None else max(self.initial_boss_health, health)
        )
        self.minimum_boss_health = (
            health if self.minimum_boss_health is None else min(self.minimum_boss_health, health)
        )
        self.boss_actor_types.update(
            int(grid[row, column, GridChannel.ACTOR_TYPE])
            for row, column in cells
            if int(grid[row, column, GridChannel.ACTOR_TYPE]) != 0
        )

    def initialize(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        position = self._position(observation, info)
        self.furthest_zone, self.furthest_floor = position[:2]
        self.boss_type = int(info.get("boss_type") or observation["player"][PlayerFeature.TASK])
        self._last_position = position
        self._floor = position[:2]
        self._visited_positions.add(position)
        self._inventory_quantities = self._inventory_counts(observation)
        self._seen_item_types = set(self._inventory_quantities)
        self.natural_prefix = dict(info.get("learning_segment") or {})
        self._observe_boss_state(observation)
        if self._has_visible_stairs(observation):
            self._stairs_seen_on_floor = True
            self.staircase_discoveries = 1
            self._pending_stair_turn = 0

    def observe(
        self,
        observation: dict[str, np.ndarray],
        reward: float,
        info: dict[str, Any],
        action: int,
    ) -> None:
        self.episode_return += float(reward)
        self.extrinsic_return += float(info.get("extrinsic_reward", 0.0))
        self.shaping_return += float(info.get("shaping_reward", 0.0))
        self.turns += 1
        self.furthest_zone, self.furthest_floor = deeper_level(
            (self.furthest_zone, self.furthest_floor),
            (int(info.get("zone") or 0), int(info.get("floor") or 0)),
        )
        self.boss_type = max(
            self.boss_type,
            int(info.get("boss_type") or observation["player"][PlayerFeature.TASK]),
        )
        self.max_gold = max(
            self.max_gold,
            int(observation["player"][PlayerFeature.GOLD]),
        )
        inventory = self._inventory_counts(observation)
        self.item_pickups += sum(
            max(quantity - self._inventory_quantities.get(item_type, 0), 0)
            for item_type, quantity in inventory.items()
        )
        self._collected_item_types.update(set(inventory) - self._seen_item_types)
        self._seen_item_types.update(inventory)
        self._inventory_quantities = inventory
        self._observe_boss_state(observation)
        if 0 <= action < ACTION_COUNT:
            self.action_counts[action] += 1
        position = self._position(observation, info)
        current_floor = position[:2]
        if current_floor != self._floor:
            outcome = info.get("action_outcome") or {}
            descent_source = outcome.get("descent_source")
            if descent_source == "stairs":
                self.staircase_exits += 1
                if self._pending_stair_turn is not None:
                    self.stair_discovery_to_exit_turns.append(self.turns - self._pending_stair_turn)
            elif descent_source == "trapdoor":
                self.trapdoor_descents += 1
            elif descent_source == "unknown":
                self.unknown_descents += 1
            elif self._pending_stair_turn is not None:
                # Backward compatibility for synthetic/legacy observations
                # that predate action-outcome descent attribution.
                self.staircase_exits += 1
                self.stair_discovery_to_exit_turns.append(self.turns - self._pending_stair_turn)
            self._floor = current_floor
            self._stairs_seen_on_floor = False
            self._pending_stair_turn = None
        events = list(info.get("raw_events", []))
        event_kinds = {str(event.get("kind", "")) for event in events}
        outcome_category = str((info.get("action_outcome") or {}).get("category", ""))
        if outcome_category:
            self.action_outcome_counts[outcome_category] = (
                self.action_outcome_counts.get(outcome_category, 0) + 1
            )
        contract = dict(info.get("action_contract") or {})
        self.known_invalid_wall_discoveries += int(
            bool(contract.get("newly_learned_invalid_wall", False))
        )
        self.masked_direction_observations += int(contract.get("masked_direction_count", 0) or 0)
        self.effective_masked_direction_observations += int(
            contract.get("effective_masked_direction_count", 0) or 0
        )
        self.navigation_prior_turns += int(bool(contract.get("navigation_prior_active", False)))
        self.navigation_masked_direction_observations += len(
            contract.get("navigation_masked_directions", []) or []
        )
        self.max_remembered_wall_states = max(
            self.max_remembered_wall_states,
            int(contract.get("remembered_wall_states", 0) or 0),
        )
        self.max_remembered_hazards = max(
            self.max_remembered_hazards,
            int(contract.get("remembered_hazards", 0) or 0),
        )
        unchanged = position == self._last_position
        if unchanged:
            self.unchanged_position_turns += 1
            self._unchanged_position_streak += 1
            self.max_unchanged_position_streak = max(
                self.max_unchanged_position_streak,
                self._unchanged_position_streak,
            )
        else:
            self._unchanged_position_streak = 0
        if action == int(Action.WAIT) and unchanged and not events:
            self.idle_turns += 1
        if unchanged and event_kinds & {"enemy_damage", "enemy_kill"}:
            self.productive_stationary_combat_turns += 1
            self._last_unchanged_direction = None
            self._repeated_direction_streak = 0
        elif unchanged and event_kinds & {
            "item_collected",
            "container_opened",
            "currency_collected",
        }:
            self.productive_stationary_interaction_turns += 1
            self._last_unchanged_direction = None
            self._repeated_direction_streak = 0
        elif unchanged and int(Action.UP) <= action <= int(Action.LEFT):
            self.unchanged_direction_turns += 1
            if action == self._last_unchanged_direction:
                self.repeated_direction_turns += 1
                self._repeated_direction_streak += 1
            else:
                self._repeated_direction_streak = 1
            self.max_repeated_direction_streak = max(
                self.max_repeated_direction_streak, self._repeated_direction_streak
            )
            self._last_unchanged_direction = action
        elif unchanged and action != int(Action.WAIT) and not events:
            self.special_no_effect_turns += 1
        else:
            self._last_unchanged_direction = None
            self._repeated_direction_streak = 0
        self._last_position = position
        self._visited_positions.add(position)
        if self._has_visible_stairs(observation) and not self._stairs_seen_on_floor:
            self._stairs_seen_on_floor = True
            self.staircase_discoveries += 1
            self._pending_stair_turn = self.turns
        for event in events:
            kind = str(event.get("kind", ""))
            amount = int(event.get("amount", 0) or 0)
            data = dict(event.get("data") or {})
            is_boss = bool(data.get("boss", False))
            is_boss_add = bool(data.get("boss_add", False))
            if kind == "enemy_kill":
                self.enemy_kills += max(amount, 1)
                self.boss_kills += int(is_boss)
                self.boss_add_kills += int(is_boss_add)
            elif kind == "currency_collected":
                self.currency_pickups += 1
                self.currency_value += amount
            elif kind == "enemy_damage":
                self.enemy_damage += amount
                if is_boss:
                    self.boss_damage += amount
                    actor_type = int(data.get("actor_type", 0) or 0)
                    if actor_type:
                        self.boss_actor_types.add(actor_type)
                    if self.initial_boss_health is not None:
                        inferred_health = max(
                            self.initial_boss_health - self.boss_damage,
                            0,
                        )
                        self.minimum_boss_health = (
                            inferred_health
                            if self.minimum_boss_health is None
                            else min(self.minimum_boss_health, inferred_health)
                        )
                if is_boss_add:
                    self.boss_add_damage += amount
            elif kind == "player_damage":
                self.player_damage += amount

    def finish(self, status: str) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "worker_id": self.worker_id,
            "run_id": self.run_id,
            "episode_return": self.episode_return,
            "extrinsic_return": self.extrinsic_return,
            "shaping_return": self.shaping_return,
            "natural_prefix": self.natural_prefix,
            "turns": self.turns,
            "furthest_zone": self.furthest_zone,
            "furthest_floor": self.furthest_floor,
            "boss_type": self.boss_type,
            "max_gold": self.max_gold,
            "enemy_kills": self.enemy_kills,
            "item_pickups": self.item_pickups,
            "unique_item_types": len(self._collected_item_types),
            "currency_pickups": self.currency_pickups,
            "currency_value": self.currency_value,
            "enemy_damage": self.enemy_damage,
            "boss_damage": self.boss_damage,
            "boss_add_damage": self.boss_add_damage,
            "boss_kills": self.boss_kills,
            "boss_add_kills": self.boss_add_kills,
            "boss_actor_types": sorted(self.boss_actor_types),
            "boss_phase_depth": min(len(self.boss_actor_types), 4),
            "initial_boss_health": self.initial_boss_health,
            "minimum_boss_health": self.minimum_boss_health,
            "death_metal_phase4_reached": bool(
                self.boss_type == int(BossType.DEATH_METAL)
                and len(self.boss_actor_types) >= 4
                and self.minimum_boss_health is not None
                and self.minimum_boss_health <= 2
                and self.boss_damage >= 7
            ),
            "player_damage": self.player_damage,
            "action_counts": self.action_counts,
            "wait_actions": self.action_counts[int(Action.WAIT)],
            "unchanged_position_turns": self.unchanged_position_turns,
            "max_unchanged_position_streak": self.max_unchanged_position_streak,
            "idle_turns": self.idle_turns,
            "productive_stationary_combat_turns": self.productive_stationary_combat_turns,
            "productive_stationary_interaction_turns": (
                self.productive_stationary_interaction_turns
            ),
            "unchanged_direction_turns": self.unchanged_direction_turns,
            "repeated_direction_turns": self.repeated_direction_turns,
            "special_no_effect_turns": self.special_no_effect_turns,
            "action_outcome_counts": self.action_outcome_counts,
            "known_invalid_wall_discoveries": self.known_invalid_wall_discoveries,
            "masked_direction_observations": self.masked_direction_observations,
            "effective_masked_direction_observations": (
                self.effective_masked_direction_observations
            ),
            "navigation_prior_turns": self.navigation_prior_turns,
            "navigation_masked_direction_observations": (
                self.navigation_masked_direction_observations
            ),
            "max_remembered_wall_states": self.max_remembered_wall_states,
            "max_remembered_hazards": self.max_remembered_hazards,
            "max_repeated_direction_streak": self.max_repeated_direction_streak,
            "unique_positions": len(self._visited_positions),
            "staircase_discoveries": self.staircase_discoveries,
            "staircase_exits": self.staircase_exits,
            "trapdoor_descents": self.trapdoor_descents,
            "unknown_descents": self.unknown_descents,
            "stair_discovery_to_exit_turns": self.stair_discovery_to_exit_turns,
            "status": status,
        }


def masked_random_actions(action_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    actions: list[int] = []
    for mask in action_mask:
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise RuntimeError("Live observation contains no legal action")
        actions.append(int(rng.choice(legal)))
    return np.asarray(actions, dtype=np.int64)


def summarize_episodes(episodes: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if not episodes:
        raise ValueError("At least one episode is required")
    count = len(episodes)
    progress = [
        level_progress(int(episode["furthest_zone"]), int(episode["furthest_floor"]))
        for episode in episodes
    ]
    deepest_episode = max(
        episodes,
        key=lambda episode: level_progress(
            int(episode["furthest_zone"]), int(episode["furthest_floor"])
        ),
    )
    furthest_level = max(progress)
    total_turns = sum(int(episode["turns"]) for episode in episodes)
    action_counts = [
        sum(int(episode.get("action_counts", [0] * ACTION_COUNT)[index]) for episode in episodes)
        for index in range(ACTION_COUNT)
    ]
    stair_delays = [
        int(delay)
        for episode in episodes
        for delay in episode.get("stair_discovery_to_exit_turns", [])
    ]
    staircase_discoveries = sum(
        int(episode.get("staircase_discoveries", 0)) for episode in episodes
    )
    staircase_exits = sum(int(episode.get("staircase_exits", 0)) for episode in episodes)
    trapdoor_descents = sum(int(episode.get("trapdoor_descents", 0)) for episode in episodes)
    unknown_descents = sum(int(episode.get("unknown_descents", 0)) for episode in episodes)
    outcome_names = sorted(
        {
            str(name)
            for episode in episodes
            for name in dict(episode.get("action_outcome_counts", {}))
        }
    )
    action_outcome_counts = {
        name: sum(
            int(dict(episode.get("action_outcome_counts", {})).get(name, 0)) for episode in episodes
        )
        for name in outcome_names
    }
    boss_type_counts = {
        str(boss_type): sum(int(episode.get("boss_type", 0)) == boss_type for episode in episodes)
        for boss_type in sorted({int(episode.get("boss_type", 0)) for episode in episodes})
    }
    death_metal_episodes = [
        episode
        for episode in episodes
        if int(episode.get("boss_type", 0)) == int(BossType.DEATH_METAL)
    ]
    prefixes = [
        dict(episode.get("natural_prefix") or {})
        for episode in episodes
        if episode.get("natural_prefix")
    ]
    return {
        "policy": policy,
        "episodes": count,
        "completion_rate": sum(episode["status"] == "won" for episode in episodes) / count,
        "curriculum_completion_rate": sum(
            episode["status"] == "curriculum_complete" for episode in episodes
        )
        / count,
        "death_rate": sum(episode["status"] == "dead" for episode in episodes) / count,
        "abort_rate": sum(episode["status"] == "aborted" for episode in episodes) / count,
        "step_limit_rate": sum(episode["status"] == "step_limit" for episode in episodes) / count,
        "mean_return": float(np.mean([episode["episode_return"] for episode in episodes])),
        "mean_extrinsic_return": float(
            np.mean([episode.get("extrinsic_return", 0.0) for episode in episodes])
        ),
        "mean_shaping_return": float(
            np.mean([episode.get("shaping_return", 0.0) for episode in episodes])
        ),
        "mean_turns": float(np.mean([episode["turns"] for episode in episodes])),
        "mean_progress": float(np.mean(progress)),
        # ``furthest_floor`` historically stores the one-based sequential level.
        # Keep it for report/checkpoint compatibility and expose unambiguous names.
        "furthest_zone": int(deepest_episode["furthest_zone"]),
        "furthest_floor": furthest_level,
        "furthest_level": furthest_level,
        "deepest_zone": int(deepest_episode["furthest_zone"]),
        "deepest_floor": int(deepest_episode["furthest_floor"]),
        "boss_type_counts": boss_type_counts,
        "natural_prefix_episodes": len(prefixes),
        "natural_prefix_acquisition_rate": (
            float(np.mean([bool(prefix.get("acquired", False)) for prefix in prefixes]))
            if prefixes
            else 0.0
        ),
        "natural_prefix_mean_guide_turns": (
            float(np.mean([int(prefix.get("guide_turns", 0)) for prefix in prefixes]))
            if prefixes
            else 0.0
        ),
        "natural_prefix_mean_attempts": (
            float(np.mean([int(prefix.get("attempts", 0)) for prefix in prefixes]))
            if prefixes
            else 0.0
        ),
        "natural_prefix_boundaries_valid": bool(prefixes)
        and all(
            not bool(prefix.get("acquired", False))
            or bool((prefix.get("boundary") or {}).get("reached"))
            for prefix in prefixes
        ),
        "mean_max_gold": float(np.mean([episode["max_gold"] for episode in episodes])),
        "enemy_kills": sum(int(episode["enemy_kills"]) for episode in episodes),
        "item_pickups": sum(int(episode["item_pickups"]) for episode in episodes),
        "unique_item_types": sum(int(episode.get("unique_item_types", 0)) for episode in episodes),
        "repeat_item_transactions": sum(
            max(
                int(episode.get("item_pickups", 0)) - int(episode.get("unique_item_types", 0)),
                0,
            )
            for episode in episodes
        ),
        "currency_pickups": sum(int(episode.get("currency_pickups", 0)) for episode in episodes),
        "currency_value": sum(int(episode.get("currency_value", 0)) for episode in episodes),
        "enemy_damage": sum(int(episode["enemy_damage"]) for episode in episodes),
        "boss_damage": sum(int(episode.get("boss_damage", 0)) for episode in episodes),
        "boss_add_damage": sum(int(episode.get("boss_add_damage", 0)) for episode in episodes),
        "boss_kills": sum(int(episode.get("boss_kills", 0)) for episode in episodes),
        "boss_add_kills": sum(int(episode.get("boss_add_kills", 0)) for episode in episodes),
        "death_metal_phase4_rate": (
            sum(
                bool(episode.get("death_metal_phase4_reached", False))
                for episode in death_metal_episodes
            )
            / len(death_metal_episodes)
            if death_metal_episodes
            else 0.0
        ),
        "mean_boss_phase_depth": float(
            np.mean([int(episode.get("boss_phase_depth", 0)) for episode in episodes])
        ),
        "player_damage": sum(int(episode["player_damage"]) for episode in episodes),
        "action_counts": action_counts,
        "wait_rate": sum(int(episode.get("wait_actions", 0)) for episode in episodes)
        / max(total_turns, 1),
        "unchanged_position_rate": sum(
            int(episode.get("unchanged_position_turns", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "idle_rate": sum(int(episode.get("idle_turns", 0)) for episode in episodes)
        / max(total_turns, 1),
        "productive_stationary_combat_rate": sum(
            int(episode.get("productive_stationary_combat_turns", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "productive_stationary_interaction_rate": sum(
            int(episode.get("productive_stationary_interaction_turns", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "unchanged_direction_rate": sum(
            int(episode.get("unchanged_direction_turns", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "repeated_direction_rate": sum(
            int(episode.get("repeated_direction_turns", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "special_no_effect_rate": sum(
            int(episode.get("special_no_effect_turns", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "action_outcome_counts": action_outcome_counts,
        "wall_attempt_rate": action_outcome_counts.get("wall_attempt", 0) / max(total_turns, 1),
        "known_invalid_wall_discoveries": sum(
            int(episode.get("known_invalid_wall_discoveries", 0)) for episode in episodes
        ),
        "mean_masked_directions": sum(
            int(episode.get("masked_direction_observations", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "mean_effective_masked_directions": sum(
            int(episode.get("effective_masked_direction_observations", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "navigation_prior_rate": sum(
            int(episode.get("navigation_prior_turns", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "mean_navigation_masked_directions": sum(
            int(episode.get("navigation_masked_direction_observations", 0)) for episode in episodes
        )
        / max(total_turns, 1),
        "mean_max_remembered_wall_states": float(
            np.mean([episode.get("max_remembered_wall_states", 0) for episode in episodes])
        ),
        "mean_max_remembered_hazards": float(
            np.mean([episode.get("max_remembered_hazards", 0) for episode in episodes])
        ),
        "mean_max_repeated_direction_streak": float(
            np.mean([episode.get("max_repeated_direction_streak", 0) for episode in episodes])
        ),
        "mean_max_unchanged_position_streak": float(
            np.mean([episode.get("max_unchanged_position_streak", 0) for episode in episodes])
        ),
        "unique_positions_per_100_turns": 100.0
        * sum(int(episode.get("unique_positions", 0)) for episode in episodes)
        / max(total_turns, 1),
        "staircase_discovery_rate": sum(
            int(episode.get("staircase_discoveries", 0)) > 0 for episode in episodes
        )
        / count,
        "staircase_discoveries": staircase_discoveries,
        "staircase_exits": staircase_exits,
        "trapdoor_descents": trapdoor_descents,
        "unknown_descents": unknown_descents,
        "staircase_conversion_rate": staircase_exits / max(staircase_discoveries, 1),
        "mean_stair_discovery_to_exit_turns": (
            float(np.mean(stair_delays)) if stair_delays else None
        ),
        "results": episodes,
    }


def compare_summaries(reference: dict[str, Any], trained: dict[str, Any]) -> dict[str, float]:
    fields = (
        "completion_rate",
        "death_rate",
        "mean_return",
        "mean_turns",
        "mean_progress",
        "mean_max_gold",
        "enemy_kills",
        "item_pickups",
        "enemy_damage",
        "boss_damage",
        "boss_add_damage",
        "boss_kills",
        "boss_add_kills",
        "player_damage",
    )
    return {f"{field}_delta": float(trained[field]) - float(reference[field]) for field in fields}


def _model_actions(
    model: PolicyModel,
    observation: dict[str, np.ndarray],
    hidden: Tensor,
    device: torch.device,
    previous_actions: np.ndarray,
    previous_rewards: np.ndarray,
) -> tuple[np.ndarray, Tensor]:
    tensors = {key: torch.from_numpy(value).to(device) for key, value in observation.items()}
    tensors["previous_action"] = torch.from_numpy(previous_actions).to(device)
    tensors["previous_reward"] = torch.from_numpy(previous_rewards).to(device)
    with torch.inference_mode():
        actions, _, _, _, next_hidden = model.act(tensors, hidden, deterministic=True)
    return actions.cpu().numpy(), next_hidden


def zero_hidden_rows(hidden: Tensor, indices: list[int]) -> Tensor:
    """Reset selected recurrent slots without mutating inference tensors."""
    keep = torch.ones(hidden.shape[0], dtype=hidden.dtype, device=hidden.device)
    keep[indices] = 0
    return hidden * keep.reshape(hidden.shape[0], *([1] * (hidden.ndim - 1)))


def recurrent_state_for_action(
    model: PolicyModel,
    hidden: Tensor,
    mode: str,
    *,
    device: torch.device,
) -> Tensor:
    """Select carried memory or a fresh state for a single policy decision."""
    if mode in {"carry", "reset-on-floor-transition"}:
        return hidden
    if mode == "reset-every-step":
        return model.initial_state(1, device=device)
    raise ValueError(f"Unknown recurrent-state mode: {mode}")


def recurrent_state_after_transition(
    model: PolicyModel,
    next_hidden: Tensor,
    mode: str,
    *,
    previous_level: tuple[int, int],
    next_level: tuple[int, int],
    device: torch.device,
) -> Tensor:
    """Optionally clear temporal memory at an observed floor boundary."""
    if mode not in RECURRENT_STATE_MODES:
        raise ValueError(f"Unknown recurrent-state mode: {mode}")
    if mode == "reset-on-floor-transition" and next_level != previous_level:
        return model.initial_state(1, device=device)
    return next_hidden


def evaluate_live_policy(
    environment: AutoDancerVectorEnv,
    *,
    seeds: list[int],
    max_steps: int,
    policy_seed: int,
    device: torch.device,
    model: PolicyModel | None = None,
    dashboard_state: DashboardState | None = None,
    action_contract: str = "current",
    policy_mode: str = "deterministic",
    recurrent_state_mode: str = "carry",
    guide_model: PolicyModel | None = None,
    natural_prefix: NaturalPrefixConfig | None = None,
) -> list[dict[str, Any]]:
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if policy_mode not in ("deterministic", "stochastic"):
        raise ValueError("policy_mode must be deterministic or stochastic")
    if model is not None:
        return _evaluate_model_async(
            environment,
            model,
            seeds=seeds,
            max_steps=max_steps,
            policy_seed=policy_seed,
            device=device,
            dashboard_state=dashboard_state,
            action_contract=action_contract,
            deterministic=policy_mode == "deterministic",
            recurrent_state_mode=recurrent_state_mode,
            guide_model=guide_model,
            natural_prefix=natural_prefix,
        )
    if natural_prefix is not None or guide_model is not None:
        raise ValueError("natural-prefix evaluation requires a trained learner model")
    rng = np.random.default_rng(policy_seed)
    results: list[dict[str, Any]] = []
    parking_seed = 2_000_000_000
    contract_memory = ActionContractMemory(action_contract, environment.num_envs)

    for start in range(0, len(seeds), environment.num_envs):
        wave_seeds = seeds[start : start + environment.num_envs]
        active_count = len(wave_seeds)
        reset_seeds = [*wave_seeds]
        if active_count < environment.num_envs:
            padding_count = environment.num_envs - active_count
            reset_seeds.extend(range(parking_seed, parking_seed + padding_count))
            parking_seed += padding_count
        observation, infos = environment.reset(reset_seeds)
        observation = contract_memory.reset_batch(observation)
        if dashboard_state is not None:
            dashboard_state.update_workers(environment.worker_ids, observation, infos)
        accumulators: list[EpisodeAccumulator | None] = [
            (
                EpisodeAccumulator(
                    seed=int(
                        infos[index].get("seed")
                        if infos[index].get("seed") is not None
                        else wave_seeds[index]
                    ),
                    worker_id=environment.worker_ids[index],
                    run_id=str(infos[index].get("run_id", "")),
                    furthest_zone=int(infos[index].get("zone") or 0),
                    furthest_floor=int(infos[index].get("floor") or 0),
                    max_gold=int(observation["player"][index, PlayerFeature.GOLD]),
                )
                if index < active_count
                else None
            )
            for index in range(environment.num_envs)
        ]
        for index, accumulator in enumerate(accumulators):
            if accumulator is not None:
                accumulator.initialize(
                    {key: value[index] for key, value in observation.items()},
                    infos[index],
                )
        hidden = (
            model.initial_state(environment.num_envs, device=device) if model is not None else None
        )
        previous_actions = np.full(environment.num_envs, START_ACTION, dtype=np.int64)
        previous_rewards = np.zeros(environment.num_envs, dtype=np.float32)

        while any(accumulator is not None for accumulator in accumulators):
            if model is None:
                actions = masked_random_actions(observation["action_mask"], rng)
                next_hidden = None
            else:
                assert hidden is not None
                actions, next_hidden = _model_actions(
                    model, observation, hidden, device, previous_actions, previous_rewards
                )
            raw_next_observation, rewards, terminated, truncated, step_infos = environment.step(
                actions
            )
            for index in range(environment.num_envs):
                step_infos[index]["action_contract"] = contract_memory.observe(
                    index,
                    {key: value[index] for key, value in observation.items()},
                    int(actions[index]),
                    {key: value[index] for key, value in raw_next_observation.items()},
                    step_infos[index],
                )
            next_observation = contract_memory.apply_batch(raw_next_observation)
            if dashboard_state is not None:
                dashboard_state.update_workers(
                    environment.worker_ids,
                    next_observation,
                    step_infos,
                    actions=actions,
                    rewards=rewards,
                )
            reset_indices: list[int] = []
            for index, accumulator in enumerate(accumulators):
                done = bool(terminated[index] or truncated[index])
                if accumulator is not None:
                    accumulator.observe(
                        {key: value[index] for key, value in next_observation.items()},
                        float(rewards[index]),
                        step_infos[index],
                        int(actions[index]),
                    )
                    reached_limit = accumulator.turns >= max_steps and not done
                    if done or reached_limit:
                        status = (
                            str(step_infos[index].get("episode_status", "aborted"))
                            if done
                            else "step_limit"
                        )
                        results.append(accumulator.finish(status))
                        accumulators[index] = None
                        reset_indices.append(index)
                elif done:
                    reset_indices.append(index)

            if not any(accumulator is not None for accumulator in accumulators):
                break
            if reset_indices:
                reset_seeds = list(range(parking_seed, parking_seed + len(reset_indices)))
                parking_seed += len(reset_indices)
                reset_results = environment.reset_at(reset_indices, reset_seeds)
                effective_resets = [
                    contract_memory.reset_slot(index, result[0])
                    for index, result in zip(reset_indices, reset_results, strict=True)
                ]
                replace_observation_rows(
                    next_observation,
                    reset_indices,
                    effective_resets,
                )
                if next_hidden is not None:
                    next_hidden = zero_hidden_rows(next_hidden, reset_indices)
            observation = next_observation
            hidden = next_hidden
            previous_actions = actions.astype(np.int64, copy=True)
            previous_rewards = rewards.astype(np.float32, copy=True)
            previous_actions[reset_indices] = START_ACTION
            previous_rewards[reset_indices] = 0.0
    return results


def _evaluation_natural_prefix(
    *,
    worker: Any,
    worker_id: str,
    slot: int,
    seed: int,
    observation: dict[str, np.ndarray],
    info: dict[str, Any],
    guide_model: PolicyModel,
    learner_model: PolicyModel,
    guide_scheduler: InferenceScheduler,
    learner_scheduler: InferenceScheduler,
    config: NaturalPrefixConfig,
    device: torch.device,
    contract_memory: ActionContractMemory,
    dashboard_state: DashboardState | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Tensor, int, float]:
    """Acquire the same legal guide boundary used by PPO collection."""

    total_turns = 0
    failures: list[dict[str, Any]] = []
    for attempt in range(config.max_attempts):
        tracker = DeathMetalPhaseTracker(config)
        tracker.observe(observation, info)
        guide_hidden = guide_model.initial_state(1, device=device)
        learner_hidden = learner_model.initial_state(1, device=device)
        previous_action = START_ACTION
        previous_reward = 0.0
        last_action = START_ACTION
        last_reward = 0.0
        for guide_turn in range(config.max_guide_turns):
            sample = float(
                np.random.default_rng(
                    np.random.SeedSequence(
                        [config.guide_policy_seed, seed, slot, attempt, guide_turn]
                    )
                ).random()
            )
            action, _, _, next_guide_hidden = guide_scheduler.infer(
                observation,
                previous_action,
                previous_reward,
                guide_hidden,
                sample,
            )
            next_learner_hidden = learner_hidden
            if config.recurrent_state_mode == "warm":
                _, _, _, next_learner_hidden = learner_scheduler.infer(
                    observation,
                    previous_action,
                    previous_reward,
                    learner_hidden,
                    0.5,
                )
            next_observation, reward, terminated, truncated, next_info = worker.step(action)
            total_turns += 1
            next_info = dict(next_info)
            next_info["natural_prefix_stage"] = "guide"
            next_info["natural_prefix_attempt"] = attempt + 1
            next_info["natural_prefix_guide_turn"] = guide_turn + 1
            tracker.observe(next_observation, next_info)
            if dashboard_state is not None:
                dashboard_state.update_worker(
                    slot,
                    worker_id,
                    next_observation,
                    next_info,
                    action=action,
                    reward=float(reward),
                )
            last_action = action
            last_reward = float(reward)
            if tracker.reached and not terminated and not truncated:
                metadata = {
                    **config.specification(),
                    "acquired": True,
                    "attempts": attempt + 1,
                    "failed_attempts": len(failures),
                    "guide_turns": total_turns,
                    "handoff_sequence": int(next_info.get("sequence", -1)),
                    "handoff_run_id": str(next_info.get("run_id", "")),
                    "handoff_seed": int(next_info.get("seed", seed)),
                    "boundary": tracker.snapshot(),
                }
                handed_observation, handed_info = worker.begin_learning_segment(
                    next_info,
                    metadata=metadata,
                )
                effective = contract_memory.reset_slot(slot, handed_observation)
                if config.recurrent_state_mode == "warm":
                    return effective, handed_info, next_learner_hidden, last_action, last_reward
                return (
                    effective,
                    handed_info,
                    learner_model.initial_state(1, device=device),
                    START_ACTION,
                    0.0,
                )
            observation = next_observation
            info = next_info
            guide_hidden = next_guide_hidden
            learner_hidden = next_learner_hidden
            previous_action = action
            previous_reward = float(reward)
            if terminated or truncated:
                break
        failures.append(
            {
                "attempt": attempt + 1,
                "status": str(info.get("episode_status", "guide_limit")),
                "boundary": tracker.snapshot(),
            }
        )
        if attempt + 1 < config.max_attempts:
            observation, info = worker.reset(seed=seed)
    raise NaturalPrefixError(
        worker_id,
        config,
        failures=failures,
        guide_turns=total_turns,
        observation=observation,
        info=info,
    )


def _evaluate_model_async(
    environment: AutoDancerVectorEnv,
    model: PolicyModel,
    *,
    seeds: list[int],
    max_steps: int,
    policy_seed: int,
    device: torch.device,
    dashboard_state: DashboardState | None,
    action_contract: str,
    deterministic: bool,
    recurrent_state_mode: str = "carry",
    guide_model: PolicyModel | None = None,
    natural_prefix: NaturalPrefixConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate model slots without a barrier or timing-dependent action samples."""
    if recurrent_state_mode not in RECURRENT_STATE_MODES:
        raise ValueError(f"Unknown recurrent-state mode: {recurrent_state_mode}")
    if (guide_model is None) != (natural_prefix is None):
        raise ValueError("guide_model and natural_prefix must be supplied together")
    results: list[dict[str, Any]] = []
    parking_seed = 2_100_000_000
    model.eval()
    contract_memory = ActionContractMemory(action_contract, environment.num_envs)
    for start in range(0, len(seeds), environment.num_envs):
        wave_seeds = seeds[start : start + environment.num_envs]
        padding = environment.num_envs - len(wave_seeds)
        reset_seeds = [*wave_seeds, *range(parking_seed, parking_seed + padding)]
        parking_seed += padding
        observations, infos = environment.reset(reset_seeds)
        if natural_prefix is None:
            observations = contract_memory.reset_batch(observations)
        scheduler = InferenceScheduler(
            model,
            device=device,
            max_batch=max(len(wave_seeds), 1),
            batch_delay=0.002,
            deterministic=deterministic,
        )
        guide_scheduler = (
            None
            if guide_model is None
            else InferenceScheduler(
                guide_model,
                device=device,
                max_batch=max(len(wave_seeds), 1),
                batch_delay=0.002,
                deterministic=bool(natural_prefix and natural_prefix.deterministic_guide),
            )
        )

        def run_slot(
            index: int,
            wave_seeds: list[int] = wave_seeds,
            observations: dict[str, np.ndarray] = observations,
            infos: list[dict[str, Any]] = infos,
            scheduler: InferenceScheduler = scheduler,
            guide_scheduler: InferenceScheduler | None = guide_scheduler,
        ) -> dict[str, Any]:
            seed = wave_seeds[index]
            worker_id = environment.worker_ids[index]
            observation = {key: value[index].copy() for key, value in observations.items()}
            info = dict(infos[index])
            hidden = model.initial_state(1, device=device)
            previous_action = START_ACTION
            previous_reward = 0.0
            attempts = 0
            while True:
                if (
                    natural_prefix is not None
                    and guide_model is not None
                    and guide_scheduler is not None
                ):
                    try:
                        observation, info, hidden, previous_action, previous_reward = (
                            _evaluation_natural_prefix(
                                worker=environment.environments[worker_id],
                                worker_id=worker_id,
                                slot=index,
                                seed=seed,
                                observation=observation,
                                info=info,
                                guide_model=guide_model,
                                learner_model=model,
                                guide_scheduler=guide_scheduler,
                                learner_scheduler=scheduler,
                                config=natural_prefix,
                                device=device,
                                contract_memory=contract_memory,
                                dashboard_state=dashboard_state,
                            )
                        )
                    except (TimeoutError, NativePipeError, ProtocolError) as error:
                        attempts += 1
                        failure = environment._failure(
                            index,
                            error,
                            operation="natural_prefix_evaluation",
                            context={"seed": seed, "attempt": attempts},
                        )
                        if attempts >= 3:
                            raise
                        observation, info = environment.recover(index, seed, failure=failure)
                        hidden = model.initial_state(1, device=device)
                        previous_action = START_ACTION
                        previous_reward = 0.0
                        continue
                    except NaturalPrefixError as error:
                        accumulator = EpisodeAccumulator(
                            seed=seed,
                            worker_id=worker_id,
                            run_id=str(error.info.get("run_id", "")),
                            furthest_zone=int(error.info.get("zone") or 0),
                            furthest_floor=int(error.info.get("floor") or 0),
                            max_gold=int(error.observation["player"][PlayerFeature.GOLD]),
                        )
                        accumulator.initialize(error.observation, error.info)
                        accumulator.natural_prefix = {
                            **error.config.specification(),
                            "acquired": False,
                            "attempts": len(error.failures),
                            "guide_turns": error.guide_turns,
                            "failures": error.failures,
                            "boundary": error.failures[-1].get("boundary", {}),
                        }
                        return accumulator.finish("prefix_failed")
                # An infrastructure retry replays the same game seed and exact
                # policy sample stream from turn zero.
                accumulator = EpisodeAccumulator(
                    seed=int(info.get("seed") if info.get("seed") is not None else seed),
                    worker_id=worker_id,
                    run_id=str(info.get("run_id", "")),
                    furthest_zone=int(info.get("zone") or 0),
                    furthest_floor=int(info.get("floor") or 0),
                    max_gold=int(observation["player"][PlayerFeature.GOLD]),
                )
                accumulator.initialize(observation, info)
                try:
                    while accumulator.turns < max_steps:
                        inference_hidden = recurrent_state_for_action(
                            model,
                            hidden,
                            recurrent_state_mode,
                            device=device,
                        )
                        action, _, _, next_hidden = scheduler.infer(
                            observation,
                            previous_action,
                            previous_reward,
                            inference_hidden,
                            (
                                stochastic_policy_sample(
                                    policy_seed,
                                    seed,
                                    accumulator.turns,
                                )
                                if not deterministic
                                else 0.0
                            ),
                        )
                        raw_next_observation, reward, terminated, truncated, step_info = (
                            environment.environments[worker_id].step(action)
                        )
                        previous_level = (
                            int(observation["player"][PlayerFeature.ZONE]),
                            int(observation["player"][PlayerFeature.FLOOR]),
                        )
                        next_level = (
                            int(raw_next_observation["player"][PlayerFeature.ZONE]),
                            int(raw_next_observation["player"][PlayerFeature.FLOOR]),
                        )
                        step_info["action_contract"] = contract_memory.observe(
                            index,
                            observation,
                            action,
                            raw_next_observation,
                            step_info,
                        )
                        next_observation = contract_memory.apply_slot(index, raw_next_observation)
                        if dashboard_state is not None:
                            dashboard_state.update_worker(
                                index,
                                worker_id,
                                next_observation,
                                step_info,
                                action=action,
                                reward=float(reward),
                            )
                        accumulator.observe(next_observation, float(reward), step_info, int(action))
                        observation = next_observation
                        info = step_info
                        hidden = recurrent_state_after_transition(
                            model,
                            next_hidden,
                            recurrent_state_mode,
                            previous_level=previous_level,
                            next_level=next_level,
                            device=device,
                        )
                        previous_action = action
                        previous_reward = float(reward)
                        if terminated or truncated:
                            return accumulator.finish(
                                str(step_info.get("episode_status", "aborted"))
                            )
                    return accumulator.finish("step_limit")
                except (TimeoutError, NativePipeError, ProtocolError) as error:
                    attempts += 1
                    failure = environment._failure(
                        index,
                        error,
                        operation=(
                            "deterministic_evaluation" if deterministic else "stochastic_evaluation"
                        ),
                        context={"seed": seed, "attempt": attempts},
                    )
                    if attempts >= 3:
                        raise
                    observation, info = environment.recover(index, seed, failure=failure)
                    observation = contract_memory.reset_slot(index, observation)
                    hidden = model.initial_state(1, device=device)
                    previous_action = START_ACTION
                    previous_reward = 0.0

        try:
            with ThreadPoolExecutor(max_workers=max(len(wave_seeds), 1)) as executor:
                results.extend(executor.map(run_slot, range(len(wave_seeds))))
        finally:
            scheduler.close()
            if guide_scheduler is not None:
                guide_scheduler.close()
    return results


def stochastic_policy_sample(policy_seed: int, game_seed: int, turn: int) -> float:
    """Return a timing-independent categorical sample for one evaluation turn."""
    if turn < 0:
        raise ValueError("turn must be non-negative")
    stream = np.random.default_rng(
        np.random.SeedSequence([int(policy_seed), int(game_seed), int(turn)])
    )
    return float(stream.random())


def _evaluate_deterministic_async(
    environment: AutoDancerVectorEnv,
    model: PolicyModel,
    *,
    seeds: list[int],
    max_steps: int,
    device: torch.device,
    dashboard_state: DashboardState | None,
    action_contract: str,
    guide_model: PolicyModel | None = None,
    natural_prefix: NaturalPrefixConfig | None = None,
) -> list[dict[str, Any]]:
    """Compatibility entry point used by the controller qualification suite."""
    return _evaluate_model_async(
        environment,
        model,
        seeds=seeds,
        max_steps=max_steps,
        policy_seed=0,
        device=device,
        dashboard_state=dashboard_state,
        action_contract=action_contract,
        deterministic=True,
        guide_model=guide_model,
        natural_prefix=natural_prefix,
    )


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_baseline(arguments: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(arguments.device)
    payload = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    architecture = payload.get("architecture", {})
    expected = model_from_spec(architecture, initialize=False)
    if payload.get("architecture") != expected.architecture_spec():
        raise ValueError("Checkpoint model architecture is incompatible with the schema-9 policy")
    model = expected.to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    natural_prefix = (
        None
        if arguments.natural_prefix_guide is None
        else NaturalPrefixConfig(
            target_phase=arguments.natural_prefix_target_phase,
            max_guide_turns=arguments.natural_prefix_max_turns,
            max_attempts=arguments.natural_prefix_max_attempts,
            deterministic_guide=arguments.natural_prefix_guide_mode == "deterministic",
            guide_policy_seed=arguments.natural_prefix_policy_seed,
            recurrent_state_mode=arguments.natural_prefix_recurrent_state,
        )
    )
    guide_model: PolicyModel | None = None
    if arguments.natural_prefix_guide is not None:
        guide_payload = torch.load(
            arguments.natural_prefix_guide,
            map_location=device,
            weights_only=False,
        )
        guide_model = model_from_spec(guide_payload.get("architecture", {}), initialize=False).to(
            device
        )
        guide_model.load_state_dict(guide_payload["model"])
        guide_model.eval()
    reward_config = load_reward_config(arguments.reward_config)
    config = SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=arguments.num_instances,
        startup_timeout=arguments.startup_timeout,
        turn_timeout=arguments.turn_timeout,
        reset_timeout=arguments.reset_timeout,
        max_turns=max(arguments.max_steps + 1, 2),
        reward_config=reward_config,
        affinity_policy=arguments.affinity,
        diagnostic_root=arguments.output.parent / "controller-diagnostics",
        curriculum_start_level=arguments.curriculum_start_level,
        curriculum_target_level=arguments.curriculum_target_level,
        curriculum_profile=arguments.curriculum_profile,
    )
    dashboard_state = DashboardState() if arguments.dashboard is not None else None
    dashboard_server = (
        DashboardServer(dashboard_state, host=arguments.dashboard_host, port=arguments.dashboard)
        if dashboard_state is not None
        else None
    )
    if dashboard_server is not None:
        dashboard_server.start()
        print(json.dumps({"dashboard_url": dashboard_server.url}, sort_keys=True), flush=True)
    try:
        with AutoDancerSupervisor(config) as supervisor:
            environment = AutoDancerVectorEnv(supervisor)
            if dashboard_state is not None:
                dashboard_state.set_status("evaluating")
                dashboard_state.update_training(
                    {
                        "checkpoint": arguments.checkpoint.name,
                        "episodes_target": len(arguments.seeds),
                        "max_steps_per_episode": arguments.max_steps,
                    }
                )
            try:
                random_results = (
                    []
                    if arguments.trained_only
                    else evaluate_live_policy(
                        environment,
                        seeds=arguments.seeds,
                        max_steps=arguments.max_steps,
                        policy_seed=arguments.policy_seed,
                        device=device,
                        dashboard_state=dashboard_state,
                        action_contract=arguments.action_contract,
                    )
                )
                trained_results = evaluate_live_policy(
                    environment,
                    seeds=arguments.seeds,
                    max_steps=arguments.max_steps,
                    policy_seed=arguments.policy_seed,
                    device=device,
                    model=model,
                    dashboard_state=dashboard_state,
                    action_contract=arguments.action_contract,
                    policy_mode=arguments.policy_mode,
                    recurrent_state_mode=arguments.recurrent_state_mode,
                    guide_model=guide_model,
                    natural_prefix=natural_prefix,
                )
                restarts = sum(handle.restart_count for handle in supervisor.workers.values())
                infrastructure_events = list(environment.infrastructure_events)
            finally:
                environment.close()
    finally:
        if dashboard_state is not None:
            dashboard_state.set_status("complete")
        if dashboard_server is not None:
            dashboard_server.stop()

    reference = (
        None if arguments.trained_only else summarize_episodes(random_results, "masked_random")
    )
    trained = summarize_episodes(trained_results, f"checkpoint_{arguments.policy_mode}")
    report = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "game_version": SUPPORTED_GAME_VERSION,
        "steam_build": SUPPORTED_STEAM_BUILD,
        "character": "Bard",
        "mode": (
            "AllZonesSeeded"
            if arguments.curriculum_start_level == 1
            else "AllZonesSeededCurriculum"
        ),
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": _checkpoint_hash(arguments.checkpoint),
        "checkpoint_global_step": int(payload.get("global_step", 0)),
        "checkpoint_updates": int(payload.get("updates", 0)),
        "reward": payload.get("checkpoint_metadata", {}).get("reward"),
        "checkpoint_action_contract": payload.get("checkpoint_metadata", {}).get("action_contract"),
        "evaluation_reward": reward_config.specification(),
        "num_instances": arguments.num_instances,
        "max_steps_per_episode": arguments.max_steps,
        "seeds": arguments.seeds,
        "policy_seed": arguments.policy_seed,
        "policy_mode": arguments.policy_mode,
        "action_contract": arguments.action_contract,
        "recurrent_state_mode": arguments.recurrent_state_mode,
        "curriculum_start_level": arguments.curriculum_start_level,
        "curriculum_target_level": arguments.curriculum_target_level,
        "curriculum_profile": arguments.curriculum_profile,
        "natural_prefix": (
            None
            if natural_prefix is None
            else {
                **natural_prefix.specification(),
                "guide_checkpoint": str(arguments.natural_prefix_guide.resolve()),
                "guide_checkpoint_sha256": _checkpoint_hash(arguments.natural_prefix_guide),
            }
        ),
        "worker_restarts": restarts,
        "controller_valid": restarts == 0 and not infrastructure_events,
        "infrastructure_events": infrastructure_events,
        "reference": reference,
        "trained": trained,
        "trained_minus_reference": (
            None if reference is None else compare_summaries(reference, trained)
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(arguments.output)
    return report


def run_baseline(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run evaluation, optionally attaching it to an experiment parent in MLflow."""
    tracker: Any | None = None
    experiment_id = getattr(arguments, "experiment_id", None)
    if experiment_id is not None:
        from autodancer.experiments.provenance import sha256_file
        from autodancer.experiments.tracking import ExperimentTracker, LineageConfig

        tracker = ExperimentTracker(
            LineageConfig(
                experiment_id=experiment_id,
                arm=arguments.experiment_arm,
                trial=arguments.trial_id or arguments.checkpoint.stem,
                stage="evaluation",
                run_dir=arguments.output.parent,
                store_root=arguments.experiment_root,
                tracking_uri=arguments.mlflow_tracking_uri,
                qualification_report=arguments.controller_qualification,
            ),
            game_dir=arguments.game_dir,
            mod_dir=arguments.mod_dir,
            device=arguments.device,
            parameters={
                "seeds": arguments.seeds,
                "max_steps": arguments.max_steps,
                "num_instances": arguments.num_instances,
                "policy_seed": arguments.policy_seed,
                "policy_mode": arguments.policy_mode,
                "action_contract": arguments.action_contract,
                "recurrent_state_mode": arguments.recurrent_state_mode,
                "curriculum_start_level": arguments.curriculum_start_level,
                "curriculum_target_level": arguments.curriculum_target_level,
                "curriculum_profile": arguments.curriculum_profile,
                "trained_only": arguments.trained_only,
                "reward_lineage_version": arguments.reward_lineage_version,
                "reward_config": (
                    None if arguments.reward_config is None else str(arguments.reward_config)
                ),
                "reward_config_sha256": sha256_file(arguments.reward_config),
            },
            source_checkpoint=arguments.checkpoint,
        )
        try:
            checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
            architecture_version = f"A{int(checkpoint['architecture']['version'])}"
            reward_version = arguments.reward_lineage_version
            if reward_version is None:
                raise ValueError("Tracked evaluation requires --reward-lineage-version")
            checkpoint_reward_version = (
                checkpoint.get("checkpoint_metadata", {}).get("reward", {}).get("version")
            )
            if not reward_version.startswith(f"V{checkpoint_reward_version}"):
                raise ValueError("Reward lineage version disagrees with the checkpoint")
            tracker.validate_component_versions(
                {"architecture": architecture_version, "reward": reward_version},
                config_hashes={"reward": sha256_file(arguments.reward_config)},
            )
        except BaseException as error:
            tracker.fail(error)
            raise
    try:
        report = _run_baseline(arguments)
        if tracker is not None:
            tracker.set_resolved(
                {
                    "checkpoint_global_step": report["checkpoint_global_step"],
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "reward": report["reward"],
                    "controller_valid": report["controller_valid"],
                }
            )
            metrics = {
                f"trained.{name}": value
                for name, value in report["trained"].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            tracker.log_metrics(metrics, step=report["checkpoint_global_step"])
            tracker.complete([arguments.output], summary=metrics)
        return report
    except BaseException as error:
        if tracker is not None:
            try:
                tracker.fail(error)
            except BaseException:
                pass
        raise


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 or seed >= 2**31 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be integers in [0, 2^31)")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("evaluation seeds must be unique")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a trained Bard checkpoint with a masked-random live baseline"
    )
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, default=default_mod_dir())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--seeds", type=_parse_seeds, required=True)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=8675309)
    parser.add_argument(
        "--policy-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
        help="take argmax actions or reproducibly sample the checkpoint policy",
    )
    parser.add_argument("--reward-config", type=Path)
    parser.add_argument(
        "--reward-lineage-version",
        help="component catalog label such as V2 or V4A (required for tracked runs)",
    )
    parser.add_argument(
        "--trained-only",
        action="store_true",
        help="Evaluate only the deterministic checkpoint policy, without a random baseline",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dashboard", type=int, nargs="?", const=8765)
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--turn-timeout", type=float, default=10.0)
    parser.add_argument("--reset-timeout", type=float, default=30.0)
    parser.add_argument("--affinity", choices=("auto", "none", "spread"), default="auto")
    parser.add_argument("--action-contract", choices=ACTION_CONTRACTS, default="current")
    parser.add_argument(
        "--recurrent-state-mode",
        choices=RECURRENT_STATE_MODES,
        default="carry",
        help=(
            "carry recurrent state normally, clear it only after a floor transition, "
            "or ablate it before every policy action"
        ),
    )
    parser.add_argument(
        "--curriculum-start-level",
        type=int,
        default=1,
        help="sequential All Zones level to start from after a normal seeded reset",
    )
    parser.add_argument(
        "--curriculum-target-level",
        type=int,
        help="terminate successfully after entering this sequential level",
    )
    parser.add_argument(
        "--curriculum-profile",
        choices=CURRICULUM_PROFILES,
        default="normal",
        help="qualification-only assistance applied on the curriculum start level",
    )
    parser.add_argument("--natural-prefix-guide", type=Path)
    parser.add_argument("--natural-prefix-target-phase", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--natural-prefix-max-turns", type=int, default=512)
    parser.add_argument("--natural-prefix-max-attempts", type=int, default=8)
    parser.add_argument(
        "--natural-prefix-guide-mode",
        choices=("deterministic", "stochastic"),
        default="stochastic",
    )
    parser.add_argument("--natural-prefix-policy-seed", type=int, default=0)
    parser.add_argument(
        "--natural-prefix-recurrent-state",
        choices=NATURAL_PREFIX_RECURRENT_MODES,
        default="fresh",
    )
    parser.add_argument("--experiment-id", help="registered immutable experiment id")
    parser.add_argument("--experiment-arm", help="arm id declared in experiment.yaml")
    parser.add_argument("--trial-id", help="stable evaluation trial label")
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument(
        "--controller-qualification",
        type=Path,
        default=Path("runs/controller-qualification/qualification.json"),
    )
    arguments = parser.parse_args()
    if arguments.mod_dir is None:
        parser.error("--mod-dir is required when LOCALAPPDATA is unavailable")
    if arguments.num_instances <= 0 or arguments.max_steps <= 0:
        parser.error("--num-instances and --max-steps must be positive")
    if arguments.curriculum_start_level <= 0:
        parser.error("--curriculum-start-level must be positive")
    if arguments.curriculum_profile != "normal" and arguments.curriculum_start_level <= 1:
        parser.error("assisted --curriculum-profile requires --curriculum-start-level > 1")
    if (
        arguments.curriculum_target_level is not None
        and arguments.curriculum_target_level <= arguments.curriculum_start_level
    ):
        parser.error("--curriculum-target-level must be after --curriculum-start-level")
    if arguments.natural_prefix_guide is not None:
        if not arguments.trained_only:
            parser.error("natural-prefix evaluation requires --trained-only")
        if arguments.curriculum_start_level != 4 or arguments.curriculum_target_level != 5:
            parser.error(
                "--natural-prefix-guide requires --curriculum-start-level 4 "
                "--curriculum-target-level 5"
            )
        if arguments.natural_prefix_max_turns <= 0 or arguments.natural_prefix_max_attempts <= 0:
            parser.error("natural-prefix turn and attempt limits must be positive")
    if bool(arguments.experiment_id) != bool(arguments.experiment_arm):
        parser.error("--experiment-id and --experiment-arm must be supplied together")
    report = run_baseline(arguments)
    summary = {
        "checkpoint_global_step": report["checkpoint_global_step"],
        "reference": (
            None
            if report["reference"] is None
            else {key: value for key, value in report["reference"].items() if key != "results"}
        ),
        "trained": {key: value for key, value in report["trained"].items() if key != "results"},
        "trained_minus_reference": report["trained_minus_reference"],
        "output": str(arguments.output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
