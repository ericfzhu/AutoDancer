"""AutoDancer simulator with replay, lifecycle, and correctness extensions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import numpy as np

from autodancer.envs.sim_base import AutoDancerSimEnv as _BaseAutoDancerSimEnv
from autodancer.generator import generate_world
from autodancer.model import Actor, GameEvent


class AutoDancerSimEnv(_BaseAutoDancerSimEnv):
    """Extend the core simulator without duplicating the turn engine."""

    def _advance_level(self, zone: int, floor: int) -> None:
        old = self._require_state()
        new = generate_world(
            self._require_channels(),
            self.task,
            zone,
            floor,
            player_health=old.player.health,
            player_max_health=old.player.max_health,
            gold=old.gold,
            bombs=old.bombs,
            inventory=old.inventory,
        )
        new.turn = old.turn
        new.groove = old.groove
        new.weapon_damage = old.weapon_damage
        new.facing = old.facing
        self.state = new

    def _damage_enemy(self, enemy: Actor, amount: int, source: str) -> list[GameEvent]:
        if amount <= 0 or enemy.health <= 0:
            return []
        actual = min(enemy.health, amount)
        enemy.health = max(0, enemy.health - amount)
        return [GameEvent("enemy_damage", actual, enemy.entity_id, {"source": source})]

    def snapshot(self) -> dict[str, Any]:
        """Return canonical, replay-relevant state for traces and debugging."""
        state = self._require_state()
        return {
            "seed": self._seed,
            "task": self.task_name,
            "character": "Bard",
            "width": state.width,
            "height": state.height,
            "terrain": state.terrain.astype(int).tolist(),
            "traps": state.traps.astype(int).tolist(),
            "visible": state.visible.astype(int).tolist(),
            "explored": state.explored.astype(int).tolist(),
            "items": [
                {"kind": int(item.kind), "x": item.x, "y": item.y, "value": item.value}
                for _, item in sorted(state.items.items())
            ],
            "enemies": [
                {
                    "id": enemy.entity_id,
                    "kind": int(enemy.kind),
                    "x": enemy.x,
                    "y": enemy.y,
                    "health": enemy.health,
                    "max_health": enemy.max_health,
                    "damage": enemy.damage,
                    "move_period": enemy.move_period,
                    "facing": enemy.facing,
                    "pattern_index": enemy.pattern_index,
                    "boss": enemy.boss,
                }
                for enemy in sorted(state.enemies.values(), key=lambda item: item.entity_id)
            ],
            "player": {
                "id": state.player.entity_id,
                "kind": int(state.player.kind),
                "x": state.player.x,
                "y": state.player.y,
                "health": state.player.health,
                "max_health": state.player.max_health,
                "damage": state.player.damage,
                "facing": state.player.facing,
            },
            "stairs": list(state.stairs),
            "zone": state.zone,
            "floor": state.floor,
            "turn": state.turn,
            "gold": state.gold,
            "groove": state.groove,
            "bombs": state.bombs,
            "weapon_damage": state.weapon_damage,
            "facing": state.facing,
            "inventory": state.inventory.astype(int).tolist(),
            "active_bombs": [
                {"x": bomb.x, "y": bomb.y, "fuse": bomb.fuse}
                for bomb in state.active_bombs
            ],
            "trap_cooldowns": [
                {"x": x, "y": y, "turns": turns}
                for (x, y), turns in sorted(state.trap_cooldowns.items())
            ],
            "won": state.won,
            "dead": state.dead,
            "rng": self._require_channels().snapshot(),
        }

    @staticmethod
    def _array_digest(array: np.ndarray) -> str:
        payload = array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes()
        return hashlib.sha256(payload).hexdigest()

    def state_digest(self) -> str:
        """Hash future-relevant state without serializing every grid cell to JSON."""
        state = self._require_state()
        payload = {
            "seed": self._seed,
            "task": self.task_name,
            "terrain": self._array_digest(state.terrain),
            "traps": self._array_digest(state.traps),
            "visible": self._array_digest(state.visible),
            "explored": self._array_digest(state.explored),
            "inventory": self._array_digest(state.inventory),
            "items": [
                (int(item.kind), item.x, item.y, item.value)
                for _, item in sorted(state.items.items())
            ],
            "enemies": [
                (
                    enemy.entity_id,
                    int(enemy.kind),
                    enemy.x,
                    enemy.y,
                    enemy.health,
                    enemy.max_health,
                    enemy.damage,
                    enemy.move_period,
                    enemy.facing,
                    enemy.pattern_index,
                    enemy.boss,
                )
                for enemy in sorted(state.enemies.values(), key=lambda item: item.entity_id)
            ],
            "player": (
                state.player.x,
                state.player.y,
                state.player.health,
                state.player.max_health,
                state.player.damage,
                state.player.facing,
            ),
            "stairs": state.stairs,
            "zone": state.zone,
            "floor": state.floor,
            "turn": state.turn,
            "gold": state.gold,
            "groove": state.groove,
            "bombs": state.bombs,
            "weapon_damage": state.weapon_damage,
            "facing": state.facing,
            "active_bombs": [
                (bomb.x, bomb.y, bomb.fuse) for bomb in state.active_bombs
            ],
            "trap_cooldowns": sorted(state.trap_cooldowns.items()),
            "won": state.won,
            "dead": state.dead,
            "rng": self._require_channels().snapshot(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=int)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _info(
        self, events: Iterable[GameEvent], *, truncated: bool | None = None
    ) -> dict[str, Any]:
        state = self._require_state()
        if truncated is None:
            truncated = (
                not state.won
                and not state.dead
                and self._episode_turn >= self.task.max_turns
            )
        if state.won:
            episode_status = "won"
        elif state.dead:
            episode_status = "dead"
        elif truncated:
            episode_status = "aborted"
        else:
            episode_status = "running"
        return {
            "seed": self._seed,
            "task": self.task_name,
            "character": "Bard",
            "zone": state.zone,
            "floor": state.floor,
            "turns": self._episode_turn,
            "deaths": int(state.dead),
            "completed": int(state.won),
            "episode_status": episode_status,
            "kills": self._kills,
            "floors_completed": self._floors_completed,
            "game_score": state.gold + self._kills * 10,
            "raw_events": [event.to_dict() for event in events],
            "state_digest": self.state_digest(),
        }
