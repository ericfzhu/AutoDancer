"""Deterministic, clean-room AutoDancer simulator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import gymnasium as gym
import numpy as np

from autodancer.constants import ACTION_COUNT, DIRECTION_DELTAS, Action, ItemKind, Terrain
from autodancer.generator import generate_world
from autodancer.model import Actor, Bomb, GameEvent, WorldState
from autodancer.observation import encode_observation, observation_space, update_visibility
from autodancer.render import render_grid
from autodancer.rng import RandomChannels
from autodancer.tasks import REWARD_VALUES, TASKS, TaskSpec


class AutoDancerSimEnv(gym.Env[dict[str, np.ndarray], int]):
    """A turn-based Bard environment with task-selectable generated levels.

    The engine is intentionally data-oriented. Every turn follows these phases:
    player action, enemy action, trap and item effects, cleanup, then end-state checks.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        *,
        task: str = "all_zones",
        render_mode: str | None = None,
        reward_values: dict[str, float] | None = None,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}. Choose one of {sorted(TASKS)}")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.task_name = task
        self.task: TaskSpec = TASKS[task]
        self.task_index = tuple(TASKS).index(task)
        self.render_mode = render_mode
        self.reward_values = REWARD_VALUES | (reward_values or {})
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = observation_space()
        self.state: WorldState | None = None
        self._channels: RandomChannels | None = None
        self._seed = 0
        self._episode_turn = 0
        self._kills = 0
        self._floors_completed = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self._seed = int(seed)
        self._channels = RandomChannels(self._seed)
        zone = int((options or {}).get("zone", 1))
        floor = int((options or {}).get("floor", self.task.start_floor))
        self.state = generate_world(self._channels, self.task, zone, floor)
        self._episode_turn = 0
        self._kills = 0
        self._floors_completed = 0
        update_visibility(self.state)
        observation = encode_observation(self.state, self.task_index)
        return observation, self._info([])

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        if state.dead or state.won:
            raise RuntimeError("step() was called after the episode ended; call reset()")
        try:
            selected = Action(int(action))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Action must be an integer from 0 to {ACTION_COUNT - 1}") from error

        events: list[GameEvent] = []
        current_mask = encode_observation(state, self.task_index)["action_mask"]
        if not current_mask[selected]:
            events.append(GameEvent("invalid_action", data={"action": int(selected)}))
            selected = Action.WAIT

        state.turn += 1
        self._episode_turn += 1
        events.extend(self._player_phase(selected))
        events.extend(self._enemy_phase())
        events.extend(self._effect_phase())
        events.extend(self._cleanup_phase())
        events.extend(self._end_state_phase())

        state = self._require_state()  # A level transition can replace the state.
        revealed = update_visibility(state)
        if revealed:
            events.append(GameEvent("reveal", amount=revealed))

        terminated = state.dead or state.won
        truncated = not terminated and self._episode_turn >= self.task.max_turns
        if truncated:
            events.append(GameEvent("failure", data={"reason": "turn_limit"}))
        reward = self._calculate_reward(events)
        observation = encode_observation(state, self.task_index)
        return observation, reward, terminated, truncated, self._info(events)

    def _player_phase(self, action: Action) -> list[GameEvent]:
        state = self._require_state()
        events: list[GameEvent] = []
        if action in DIRECTION_DELTAS:
            state.facing = int(action)
            dx, dy = DIRECTION_DELTAS[action]
            target_x = state.player.x + dx
            target_y = state.player.y + dy
            if not state.in_bounds(target_x, target_y):
                return [GameEvent("move_blocked")]
            enemy = next(
                (e for e in state.enemies.values() if e.position == (target_x, target_y)), None
            )
            if enemy is not None:
                events.extend(self._damage_enemy(enemy, state.weapon_damage, "melee"))
            elif state.terrain[target_y, target_x] == Terrain.WALL:
                if 0 < target_x < state.width - 1 and 0 < target_y < state.height - 1:
                    state.terrain[target_y, target_x] = Terrain.FLOOR
                    events.append(GameEvent("wall_dug", data={"x": target_x, "y": target_y}))
                else:
                    events.append(GameEvent("move_blocked"))
            elif state.is_walkable(target_x, target_y):
                state.player.x, state.player.y = target_x, target_y
                events.append(GameEvent("player_moved", data={"x": target_x, "y": target_y}))
        elif action == Action.WAIT:
            events.append(GameEvent("wait"))
        elif action == Action.BOMB:
            state.bombs -= 1
            state.active_bombs.append(Bomb(*state.player.position))
            events.append(GameEvent("bomb_placed", data={"x": state.player.x, "y": state.player.y}))
        elif action in {Action.ITEM_1, Action.ITEM_2}:
            slot = 1 if action == Action.ITEM_1 else 2
            events.extend(self._use_item(slot))
        elif action == Action.THROW:
            events.extend(self._throw_weapon())
        elif action in {Action.SPELL_1, Action.SPELL_2}:
            slot = 4 if action == Action.SPELL_1 else 5
            state.inventory[slot, 1] = max(0, int(state.inventory[slot, 1]) - 1)
            if state.inventory[slot, 1] == 0:
                state.inventory[slot] = 0
            for enemy in list(state.enemies.values()):
                if max(abs(enemy.x - state.player.x), abs(enemy.y - state.player.y)) <= 1:
                    events.extend(self._damage_enemy(enemy, 1, "spell"))
            events.append(GameEvent("spell_used", data={"slot": slot}))
        return events

    def _enemy_phase(self) -> list[GameEvent]:
        state = self._require_state()
        events: list[GameEvent] = []
        occupied = {enemy.position for enemy in state.enemies.values() if enemy.health > 0}
        for entity_id in sorted(state.enemies):
            enemy = state.enemies[entity_id]
            if enemy.health <= 0 or state.turn % enemy.move_period:
                continue
            distance = abs(enemy.x - state.player.x) + abs(enemy.y - state.player.y)
            if distance == 1:
                state.player.health -= enemy.damage
                events.append(
                    GameEvent("player_damage", enemy.damage, enemy.entity_id, {"source": "enemy"})
                )
                continue
            old_position = enemy.position
            target = self._enemy_target(enemy)
            if target != state.player.position and (
                target in occupied or not state.is_walkable(*target)
            ):
                continue
            if target == state.player.position:
                state.player.health -= enemy.damage
                events.append(
                    GameEvent("player_damage", enemy.damage, enemy.entity_id, {"source": "enemy"})
                )
            else:
                occupied.discard(old_position)
                enemy.x, enemy.y = target
                occupied.add(target)
                events.append(GameEvent("enemy_moved", entity_id=enemy.entity_id))
        return events

    def _enemy_target(self, enemy: Actor) -> tuple[int, int]:
        state = self._require_state()
        dx = int(np.sign(state.player.x - enemy.x))
        dy = int(np.sign(state.player.y - enemy.y))
        if enemy.kind.name == "GREEN_SLIME":
            dx = 0
        elif enemy.kind.name == "BLUE_SLIME":
            dy = 0
        elif enemy.kind.name == "BAT":
            rng = self._require_channels().channel("enemy_ai")
            dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[int(rng.integers(0, 4))]
        elif abs(state.player.x - enemy.x) >= abs(state.player.y - enemy.y):
            dy = 0
        else:
            dx = 0
        return enemy.x + dx, enemy.y + dy

    def _effect_phase(self) -> list[GameEvent]:
        state = self._require_state()
        events: list[GameEvent] = []
        item = state.items.pop(state.player.position, None)
        if item is not None:
            if item.kind == ItemKind.GOLD:
                state.gold += item.value
            elif item.kind == ItemKind.FOOD:
                state.player.health = min(state.player.max_health, state.player.health + item.value)
            elif item.kind == ItemKind.BOMB:
                state.bombs += item.value
            events.append(GameEvent("item_collected", item.value, data={"item": int(item.kind)}))

        if state.traps[state.player.y, state.player.x] and state.turn % 2 == 0:
            state.player.health -= 1
            events.append(GameEvent("player_damage", 1, data={"source": "trap"}))

        remaining_bombs: list[Bomb] = []
        for bomb in state.active_bombs:
            bomb.fuse -= 1
            if bomb.fuse > 0:
                remaining_bombs.append(bomb)
                continue
            events.append(GameEvent("bomb_exploded", data={"x": bomb.x, "y": bomb.y}))
            for y in range(bomb.y - 1, bomb.y + 2):
                for x in range(bomb.x - 1, bomb.x + 2):
                    if not state.in_bounds(x, y):
                        continue
                    if 0 < x < state.width - 1 and 0 < y < state.height - 1:
                        if state.terrain[y, x] == Terrain.WALL:
                            state.terrain[y, x] = Terrain.FLOOR
                    enemy = next((e for e in state.enemies.values() if e.position == (x, y)), None)
                    if enemy is not None:
                        events.extend(self._damage_enemy(enemy, 4, "bomb"))
                    if state.player.position == (x, y):
                        state.player.health -= 4
                        events.append(GameEvent("player_damage", 4, data={"source": "bomb"}))
        state.active_bombs = remaining_bombs
        return events

    def _cleanup_phase(self) -> list[GameEvent]:
        state = self._require_state()
        events: list[GameEvent] = []
        for entity_id, enemy in list(state.enemies.items()):
            if enemy.health <= 0:
                del state.enemies[entity_id]
                self._kills += 1
                state.gold += 1
                events.append(GameEvent("enemy_kill", entity_id=entity_id))
        if state.player.health <= 0:
            state.player.health = 0
            state.dead = True
            events.append(GameEvent("failure", data={"reason": "death"}))
        return events

    def _end_state_phase(self) -> list[GameEvent]:
        state = self._require_state()
        if state.dead:
            return []
        combat_task_complete = self.task_name in {"single_enemy", "mixed_room"} and not state.enemies
        navigation_complete = self.task_name == "navigation" and state.player.position == state.stairs
        floor_complete = not state.enemies and state.player.position == state.stairs
        if not (combat_task_complete or navigation_complete or floor_complete):
            return []

        self._floors_completed += 1
        event_data = {"zone": state.zone, "floor": state.floor}
        if self.task_name in {"navigation", "single_enemy", "mixed_room", "floor"}:
            state.won = True
            event_data["task_complete"] = True
            return [GameEvent("success", data=event_data)]

        if state.floor < self.task.regular_floors:
            next_zone, next_floor = state.zone, state.floor + 1
        elif self.task.include_boss and state.floor == self.task.regular_floors:
            next_zone, next_floor = state.zone, state.floor + 1
        elif state.zone < self.task.max_zone:
            next_zone, next_floor = state.zone + 1, 1
        else:
            state.won = True
            event_data["task_complete"] = True
            return [GameEvent("success", data=event_data)]

        self._advance_level(next_zone, next_floor)
        return [
            GameEvent("success", data=event_data),
            GameEvent("level_transition", data={"zone": next_zone, "floor": next_floor}),
        ]

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
        new.weapon_damage = old.weapon_damage
        new.facing = old.facing
        self.state = new

    def _damage_enemy(self, enemy: Actor, amount: int, source: str) -> list[GameEvent]:
        if amount <= 0:
            return []
        actual = min(enemy.health, amount)
        enemy.health -= amount
        return [GameEvent("enemy_damage", actual, enemy.entity_id, {"source": source})]

    def _use_item(self, slot: int) -> list[GameEvent]:
        state = self._require_state()
        item_id, charges, value = (int(v) for v in state.inventory[slot])
        if item_id == ItemKind.FOOD:
            healed = min(value or 1, state.player.max_health - state.player.health)
            state.player.health += healed
            events = [GameEvent("item_used", healed, data={"slot": slot, "item": item_id})]
        else:
            events = [GameEvent("item_used", data={"slot": slot, "item": item_id})]
        charges -= 1
        if charges <= 0:
            state.inventory[slot] = 0
        else:
            state.inventory[slot, 1] = charges
        return events

    def _throw_weapon(self) -> list[GameEvent]:
        state = self._require_state()
        facing = Action(state.facing)
        dx, dy = DIRECTION_DELTAS[facing]
        x, y = state.player.position
        events: list[GameEvent] = [GameEvent("weapon_thrown")]
        while True:
            x, y = x + dx, y + dy
            if not state.in_bounds(x, y) or state.terrain[y, x] == Terrain.WALL:
                break
            enemy = next((e for e in state.enemies.values() if e.position == (x, y)), None)
            if enemy is not None:
                events.extend(self._damage_enemy(enemy, state.weapon_damage, "throw"))
                break
        state.weapon_damage = 0
        return events

    def _calculate_reward(self, events: Iterable[GameEvent]) -> float:
        reward = self.reward_values["turn"]
        for event in events:
            if event.kind == "success":
                reward += self.reward_values["success"]
            elif event.kind == "failure":
                reward += self.reward_values["failure"]
            elif event.kind == "enemy_damage":
                reward += self.reward_values["enemy_damage"] * event.amount
            elif event.kind == "enemy_kill":
                reward += self.reward_values["enemy_kill"]
            elif event.kind == "player_damage":
                reward += self.reward_values["player_damage"] * event.amount
            elif event.kind == "reveal":
                reward += self.reward_values["reveal"] * event.amount
        return float(reward)

    def snapshot(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "seed": self._seed,
            "task": self.task_name,
            "turn": state.turn,
            "zone": state.zone,
            "floor": state.floor,
            "player": {
                "x": state.player.x,
                "y": state.player.y,
                "health": state.player.health,
                "max_health": state.player.max_health,
                "gold": state.gold,
                "bombs": state.bombs,
            },
            "enemies": [
                {
                    "id": enemy.entity_id,
                    "kind": int(enemy.kind),
                    "x": enemy.x,
                    "y": enemy.y,
                    "health": enemy.health,
                }
                for enemy in sorted(state.enemies.values(), key=lambda item: item.entity_id)
            ],
            "stairs": list(state.stairs),
            "won": state.won,
            "dead": state.dead,
        }

    def state_digest(self) -> str:
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _info(self, events: Iterable[GameEvent]) -> dict[str, Any]:
        state = self._require_state()
        return {
            "seed": self._seed,
            "task": self.task_name,
            "zone": state.zone,
            "floor": state.floor,
            "turns": self._episode_turn,
            "deaths": int(state.dead),
            "kills": self._kills,
            "floors_completed": self._floors_completed,
            "game_score": state.gold + self._kills * 10,
            "raw_events": [event.to_dict() for event in events],
            "state_digest": self.state_digest(),
        }

    def render(self) -> np.ndarray:
        state = self._require_state()
        return render_grid(encode_observation(state, self.task_index)["grid"])

    def _require_state(self) -> WorldState:
        if self.state is None:
            raise RuntimeError("Call reset() before using the environment")
        return self.state

    def _require_channels(self) -> RandomChannels:
        if self._channels is None:
            raise RuntimeError("Call reset() before using the environment")
        return self._channels

