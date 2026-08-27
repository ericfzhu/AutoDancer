"""Gymnasium environment backed exclusively by a running NecroDancer instance."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from autodancer.constants import (
    ACTION_COUNT,
    GRID_SIZE,
    Action,
    ActorKind,
    GridChannel,
    InventoryFeature,
    PlayerFeature,
    Terrain,
)
from autodancer.curriculum import EpisodeResetSpec
from autodancer.live.bridge import (
    CURRICULUM_PROFILES,
    ActionBridge,
    BridgeCommand,
    FileCommandBridge,
)
from autodancer.live.protocol import (
    JsonlTurnSource,
    ProtocolError,
    TurnSource,
    decode_observation,
    validate_record,
)
from autodancer.memory import FloorMapMemory
from autodancer.observation import observation_space
from autodancer.outcomes import classify_action_outcome
from autodancer.progress import level_progress
from autodancer.rewards import RewardConfig, RewardTracker


class AutoDancerLiveEnv(gym.Env[dict[str, np.ndarray], int]):
    """Synchronous Python/Lua adapter for one Bard game instance."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        log_path: str | Path | None = None,
        command_path: str | Path | None = None,
        turn_source: TurnSource | None = None,
        bridge: ActionBridge | None = None,
        turn_timeout: float = 5.0,
        reset_timeout: float | None = None,
        attach_existing: bool = False,
        max_turns: int = 10000,
        instance_id: str = "worker-0000",
        session_id: str | None = None,
        launch_id: str | None = None,
        reward_config: RewardConfig | None = None,
        curriculum_start_level: int = 1,
        curriculum_target_level: int | None = None,
        curriculum_profile: str = "normal",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if turn_source is None and log_path is None:
            self._source: TurnSource | None = None
        else:
            self._source = turn_source or JsonlTurnSource(Path(log_path))  # type: ignore[arg-type]
        if bridge is None and command_path is not None:
            bridge = FileCommandBridge(command_path, instance_id=instance_id)
        self._bridge = bridge
        self.instance_id = instance_id
        self.session_id = session_id
        self.launch_id = launch_id
        self.progress_callback = progress_callback
        self.turn_timeout = float(turn_timeout)
        self.reset_timeout = float(reset_timeout if reset_timeout is not None else turn_timeout)
        self.attach_existing = bool(attach_existing)
        self.max_turns = int(max_turns)
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.curriculum_start_level = int(curriculum_start_level)
        self.curriculum_target_level = (
            None if curriculum_target_level is None else int(curriculum_target_level)
        )
        self.curriculum_profile = str(curriculum_profile)
        if self.curriculum_start_level <= 0:
            raise ValueError("curriculum_start_level must be positive")
        if self.curriculum_profile not in CURRICULUM_PROFILES:
            raise ValueError(
                "curriculum_profile must be one of " + ", ".join(CURRICULUM_PROFILES)
            )
        if self.curriculum_profile != "normal" and self.curriculum_start_level <= 1:
            raise ValueError("assisted curriculum profiles require a later start level")
        if (
            self.curriculum_target_level is not None
            and self.curriculum_target_level <= self.curriculum_start_level
        ):
            raise ValueError("curriculum_target_level must be after curriculum_start_level")
        if self.attach_existing and self.curriculum_start_level != 1:
            raise ValueError("attach_existing cannot perform a curriculum level start")
        self._default_reset_spec = EpisodeResetSpec(
            "fixed",
            self.curriculum_start_level,
            self.curriculum_target_level,
            self.curriculum_profile,
        )
        self._active_reset_spec = self._default_reset_spec
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = observation_space()
        self._last_observation: dict[str, np.ndarray] | None = None
        self._episode_steps = 0
        self._episode_done = False
        self._episode_seed: int | None = None
        self._run_id: str | None = None
        self.reward_tracker = RewardTracker(reward_config)
        self.map_memory = FloorMapMemory()

    def _dependencies(self) -> tuple[TurnSource, ActionBridge]:
        if self._source is None:
            raise RuntimeError("Set log_path or inject turn_source before reset()")
        if self._bridge is None:
            raise RuntimeError("Set command_path or inject bridge before reset()")
        return self._source, self._bridge

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        reset_spec = self._resolve_reset_spec(options)
        if self.attach_existing and reset_spec.start_level != 1:
            raise ValueError("attach_existing cannot perform a curriculum level start")
        self._active_reset_spec = reset_spec
        source, bridge = self._dependencies()
        self.map_memory.reset()
        source.reset_sequence()
        if self.attach_existing:
            record = source.read_latest(self.reset_timeout)
        else:
            selected_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
            command_started = time.monotonic()
            command = bridge.reset(selected_seed)
            deadline = time.monotonic() + self.reset_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "No AutoDancer reset record arrived within "
                        f"{self.reset_timeout:.1f} seconds"
                    )
                try:
                    record = source.read(remaining)
                    break
                except ProtocolError as error:
                    if "must start with a reset record" not in str(error):
                        raise
            self._verify_acknowledgement(record, command)
        validate_record(record)
        if not self.attach_existing and record["kind"] != "reset":
            raise ProtocolError("The first record after RESTART must have kind 'reset'")
        self._episode_steps = 0
        self._episode_done = record["episode_status"] != "running"
        self._episode_seed = int(record.get("seed", 0))
        self._run_id = str(record["run_id"])
        observation, info = self._accept_record(record)
        if reset_spec.start_level > 1:
            for target_level in range(2, reset_spec.start_level + 1):
                profile = (
                    reset_spec.profile
                    if target_level == reset_spec.start_level
                    and reset_spec.profile != "normal"
                    else None
                )
                observation, info = self._goto_level(target_level, profile)
        if not self.attach_existing:
            info["reset_latency_seconds"] = time.monotonic() - command_started
        info["curriculum_reset"] = reset_spec.as_dict()
        info["curriculum_reset_id"] = reset_spec.id
        info["curriculum_start_level"] = reset_spec.start_level
        info["curriculum_target_level"] = reset_spec.target_level
        info["curriculum_profile"] = reset_spec.profile
        self.reward_tracker.reset(observation, info)
        if self.progress_callback is not None:
            self.progress_callback(info)
        return observation, info

    def _resolve_reset_spec(self, options: dict[str, Any] | None) -> EpisodeResetSpec:
        if options is None:
            return self._default_reset_spec
        unknown = set(options) - {"curriculum"}
        if unknown:
            raise ValueError(f"unknown live reset options: {sorted(unknown)}")
        value = options.get("curriculum")
        if value is None:
            return self._default_reset_spec
        if not isinstance(value, dict):
            raise ValueError("reset option 'curriculum' must be an object")
        return EpisodeResetSpec.from_mapping(value)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("step() was called after the live episode ended; call reset()")
        source, bridge = self._dependencies()
        try:
            selected = Action(int(action))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Action must be an integer from 0 to {ACTION_COUNT - 1}") from error
        if (
            self._last_observation is not None
            and not self._last_observation["action_mask"][selected]
        ):
            raise ValueError(f"Action {selected.name} is masked in the current live state")

        previous_observation = self._last_observation
        command_started = time.monotonic()
        command = bridge.send_action(selected)
        record = source.read(self.turn_timeout)
        validate_record(record)
        if record["kind"] == "reset":
            raise ProtocolError("A reset arrived while waiting for an action result")
        self._verify_acknowledgement(record, command)
        action_latency = time.monotonic() - command_started
        if self._episode_seed is not None and int(record.get("seed", -1)) != self._episode_seed:
            raise ProtocolError(
                f"Episode seed changed from {self._episode_seed} to {record.get('seed')}"
            )
        if self._run_id is not None and str(record.get("run_id")) != self._run_id:
            raise ProtocolError("Run ID changed while awaiting an action acknowledgement")

        observation, info = self._accept_record(record)
        info["curriculum_reset"] = self._active_reset_spec.as_dict()
        info["curriculum_reset_id"] = self._active_reset_spec.id
        info["curriculum_start_level"] = self._active_reset_spec.start_level
        info["curriculum_target_level"] = self._active_reset_spec.target_level
        info["curriculum_profile"] = self._active_reset_spec.profile
        self._episode_steps += 1
        events = [dict(event) for event in record.get("events", [])]
        if previous_observation is not None:
            events = self._normalize_player_damage(
                events, previous_observation, observation
            )
            events.extend(self._inventory_pickup_events(previous_observation, observation))
        status = str(record["episode_status"])
        terminated = status in {"won", "dead"}
        truncated = status == "aborted"
        current_level = level_progress(
            int(info.get("zone") or 0), int(info.get("floor") or 0)
        )
        curriculum_complete = bool(
            self._active_reset_spec.target_level is not None
            and current_level >= self._active_reset_spec.target_level
        )
        if curriculum_complete and not terminated and not truncated:
            terminated = True
            status = "curriculum_complete"
            info["episode_status"] = status
            info["curriculum_completed"] = True
            info["curriculum_reset"] = self._active_reset_spec.as_dict()
            info["curriculum_reset_id"] = self._active_reset_spec.id
            info["curriculum_target_level"] = self._active_reset_spec.target_level
        if not terminated and not truncated and self._episode_steps >= self.max_turns:
            truncated = True
            status = "time_limit"
            info["episode_status"] = status
            info["client_turn_limit"] = self.max_turns
        info["raw_events"] = events
        if previous_observation is not None:
            info["action_outcome"] = classify_action_outcome(
                previous_observation, observation, selected, info
            ).as_dict()
        info["completed"] = int(status == "won")
        info["deaths"] = int(status == "dead")
        info["turns"] = self._episode_steps
        info["action_latency_seconds"] = action_latency
        info["max_frame_bytes"] = int(getattr(source, "max_frame_bytes", 0))
        info["last_command_status"] = getattr(source, "last_status", None)
        self._episode_done = terminated or truncated
        reward, components = self.reward_tracker.score(
            observation,
            info,
            events,
            terminated=terminated,
            truncated=truncated,
        )
        info["reward_components"] = components
        extrinsic, shaping = self.reward_tracker.split_components(components)
        info["extrinsic_reward"] = extrinsic
        info["shaping_reward"] = shaping
        if self.progress_callback is not None:
            self.progress_callback(info)
        return observation, reward, terminated, truncated, info

    @staticmethod
    def _normalize_player_damage(
        events: list[dict[str, Any]],
        before: dict[str, np.ndarray],
        after: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        """Score actual Bard health loss instead of unbounded attack magnitude."""
        damage_indices = [
            index for index, event in enumerate(events) if event.get("kind") == "player_damage"
        ]
        if not damage_indices:
            return events
        previous_health = max(int(before["player"][PlayerFeature.HEALTH]), 0)
        current_health = max(int(after["player"][PlayerFeature.HEALTH]), 0)
        actual_loss = min(max(previous_health - current_health, 0), previous_health)
        raw_damage = sum(
            max(int(events[index].get("amount", 0) or 0), 0)
            for index in damage_indices
        )
        first_index = damage_indices[0]
        normalized = dict(events[first_index])
        normalized["amount"] = actual_loss
        data = dict(normalized.get("data") or {})
        data.update({"raw_damage": raw_damage, "event_count": len(damage_indices)})
        normalized["data"] = data
        return [
            normalized if index == first_index else event
            for index, event in enumerate(events)
            if index == first_index or index not in damage_indices
        ]

    @staticmethod
    def _inventory_pickup_events(
        before: dict[str, np.ndarray], after: dict[str, np.ndarray]
    ) -> list[dict[str, Any]]:
        def counts(observation: dict[str, np.ndarray]) -> dict[int, int]:
            result: dict[int, int] = {}
            for item in observation["inventory"]:
                item_type = int(item[InventoryFeature.ITEM_TYPE])
                if item_type == 0:
                    continue
                quantity = max(int(item[InventoryFeature.QUANTITY]), 1)
                result[item_type] = result.get(item_type, 0) + quantity
            return result

        previous = counts(before)
        current = counts(after)
        return [
            {
                "kind": "item_collected",
                "amount": quantity - previous.get(item_type, 0),
                "entity_id": 0,
                "data": {"item_type": item_type, "source": "inventory_delta"},
            }
            for item_type, quantity in sorted(current.items())
            if quantity > previous.get(item_type, 0)
        ]

    def qualification_goto_level(
        self, level: int
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Load a real run-sequence level for qualification-only boundary checks."""
        return self._goto_level(level)

    def _goto_level(
        self, level: int, profile: str | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Load the next real run-sequence level without inventing a transition."""
        if self._episode_done or self._run_id is None or self._episode_seed is None:
            raise RuntimeError("goto_level() requires an active reset run")
        source, bridge = self._dependencies()
        command = (
            bridge.goto_level(level)
            if profile is None
            else bridge.goto_level(level, profile)
        )
        record = source.read(self.reset_timeout)
        validate_record(record)
        self._verify_acknowledgement(record, command)
        if str(record.get("run_id")) != self._run_id:
            raise ProtocolError("Run ID changed during a qualification level transition")
        if int(record.get("seed", -1)) != self._episode_seed:
            raise ProtocolError("Seed changed during a qualification level transition")
        observation, info = self._accept_record(record)
        return observation, info

    @staticmethod
    def _verify_acknowledgement(record: dict[str, Any], command: BridgeCommand) -> None:
        acknowledgement = record.get("bridge")
        if not isinstance(acknowledgement, dict):
            raise ProtocolError("Lua did not acknowledge the Python bridge command")
        received = [
            acknowledgement.get("kind"),
            acknowledgement.get("session_id"),
            acknowledgement.get("command_id"),
        ]
        expected = [command.kind, command.session_id, command.command_id]
        if command.kind == "ACTION":
            received.append(acknowledgement.get("requested_action"))
            expected.append(None if command.action is None else int(command.action))
            engine_action = acknowledgement.get("engine_action")
            observed_action = acknowledgement.get("observed_action")
            if engine_action is None or observed_action != engine_action:
                raise ProtocolError(
                    "Lua acknowledgement did not observe the injected engine action"
                )
        elif command.kind == "RESET":
            received.append(acknowledgement.get("seed"))
            expected.append(command.seed)
            received.append(record.get("seed"))
            expected.append(command.seed)
        elif command.kind == "GOTO":
            received.append(acknowledgement.get("target_level"))
            expected.append(command.target_level)
            if command.curriculum_profile is not None:
                received.append(acknowledgement.get("curriculum_profile"))
                expected.append(command.curriculum_profile)
        if received != expected:
            raise ProtocolError(
                f"Bridge acknowledgement mismatch: expected {expected}, received {received}"
            )

    def _adapt_observation(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        observation = decode_observation(record["observation"])
        grid = observation["grid"]
        player = observation["player"]
        player[PlayerFeature.VISIBLE_ENEMIES] = int(
            np.count_nonzero(
                (grid[..., GridChannel.ACTOR_CLASS] > ActorKind.PLAYER)
                & (grid[..., GridChannel.VISIBILITY] == 2)
            )
        )
        centre = GRID_SIZE // 2
        player[PlayerFeature.ON_STAIRS] = int(
            grid[centre, centre, GridChannel.TERRAIN_CLASS] == Terrain.STAIRS
        )
        status = str(record["episode_status"])
        player[PlayerFeature.WON] = int(status == "won")
        player[PlayerFeature.DEAD] = int(status == "dead")
        observation["map_memory"] = self.map_memory.update(
            observation,
            record["observation"].get("revealed_map"),
            record["observation"].get("map_bounds"),
            record["observation"].get("revealed_map_origin"),
        )
        return observation

    def _accept_record(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if record.get("instance_id") != self.instance_id:
            raise ProtocolError(
                f"Expected worker {self.instance_id!r}, received {record.get('instance_id')!r}"
            )
        if self.session_id is not None and record.get("session_id") != self.session_id:
            raise ProtocolError("Transition supervisor session does not match this environment")
        if self.launch_id is not None and record.get("launch_id") != self.launch_id:
            raise ProtocolError("Transition launch identity does not match this environment")
        observation = self._adapt_observation(record)
        self._last_observation = observation
        status = str(record["episode_status"])
        info = {
            "protocol_schema_version": record["schema_version"],
            "instance_id": record["instance_id"],
            "session_id": record["session_id"],
            "launch_id": record["launch_id"],
            "run_id": record["run_id"],
            "sequence": record["sequence"],
            "seed": record.get("seed"),
            "character": record.get("character"),
            "game": record["game"],
            "zone": record.get("zone"),
            "floor": record.get("floor"),
            "boss_type": int(observation["player"][PlayerFeature.TASK]),
            "episode_status": status,
            "bridge": record.get("bridge"),
            "completed": int(status == "won"),
            "deaths": int(status == "dead"),
            "raw_events": [dict(event) for event in record.get("events", [])],
            "frame_bytes": int(getattr(self._source, "last_frame_bytes", 0)),
            "max_frame_bytes": int(getattr(self._source, "max_frame_bytes", 0)),
            **record.get("metrics", {}),
        }
        return observation, info
