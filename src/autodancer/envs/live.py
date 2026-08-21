"""Gymnasium environment backed exclusively by a running NecroDancer instance."""

from __future__ import annotations

import time
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
    PlayerFeature,
    Terrain,
)
from autodancer.live.bridge import ActionBridge, BridgeCommand, FileCommandBridge
from autodancer.live.protocol import (
    JsonlTurnSource,
    ProtocolError,
    TurnSource,
    decode_observation,
    validate_record,
)
from autodancer.observation import observation_space
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
        reward_config: RewardConfig | None = None,
    ) -> None:
        if turn_source is None and log_path is None:
            self._source: TurnSource | None = None
        else:
            self._source = turn_source or JsonlTurnSource(Path(log_path))  # type: ignore[arg-type]
        if bridge is None and command_path is not None:
            bridge = FileCommandBridge(command_path, instance_id=instance_id)
        self._bridge = bridge
        self.instance_id = instance_id
        self.turn_timeout = float(turn_timeout)
        self.reset_timeout = float(reset_timeout if reset_timeout is not None else turn_timeout)
        self.attach_existing = bool(attach_existing)
        self.max_turns = int(max_turns)
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = observation_space()
        self._last_observation: dict[str, np.ndarray] | None = None
        self._episode_steps = 0
        self._episode_done = False
        self.reward_tracker = RewardTracker(reward_config)

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
        del options
        source, bridge = self._dependencies()
        source.reset_sequence()
        if self.attach_existing:
            record = source.read_latest(self.reset_timeout)
        else:
            selected_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
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
        observation, info = self._accept_record(record)
        self.reward_tracker.reset(observation, info)
        return observation, info

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

        command = bridge.send_action(selected)
        record = source.read(self.turn_timeout)
        validate_record(record)
        if record["kind"] == "reset":
            raise ProtocolError("A reset arrived while waiting for an action result")
        self._verify_acknowledgement(record, command)

        observation, info = self._accept_record(record)
        self._episode_steps += 1
        events = [dict(event) for event in record.get("events", [])]
        status = str(record["episode_status"])
        terminated = status in {"won", "dead"}
        truncated = status == "aborted"
        if not terminated and not truncated and self._episode_steps >= self.max_turns:
            truncated = True
            status = "aborted"
            events.append({"kind": "failure", "amount": 1, "data": {"reason": "client_turn_limit"}})
            info["episode_status"] = status
            info["client_turn_limit"] = self.max_turns
        info["raw_events"] = events
        info["completed"] = int(status == "won")
        info["deaths"] = int(status == "dead")
        info["turns"] = self._episode_steps
        self._episode_done = terminated or truncated
        reward, components = self.reward_tracker.score(
            observation,
            info,
            events,
            terminated=terminated,
            truncated=truncated,
        )
        info["reward_components"] = components
        return observation, reward, terminated, truncated, info

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
        elif command.kind == "RESET":
            received.append(acknowledgement.get("seed"))
            expected.append(command.seed)
            received.append(record.get("seed"))
            expected.append(command.seed)
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
        player[PlayerFeature.TASK] = 0
        status = str(record["episode_status"])
        player[PlayerFeature.WON] = int(status == "won")
        player[PlayerFeature.DEAD] = int(status == "dead")
        return observation

    def _accept_record(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if record.get("instance_id") != self.instance_id:
            raise ProtocolError(
                f"Expected worker {self.instance_id!r}, received {record.get('instance_id')!r}"
            )
        observation = self._adapt_observation(record)
        self._last_observation = observation
        status = str(record["episode_status"])
        info = {
            "protocol_schema_version": record["schema_version"],
            "instance_id": record["instance_id"],
            "run_id": record["run_id"],
            "sequence": record["sequence"],
            "seed": record.get("seed"),
            "character": record.get("character"),
            "game": record["game"],
            "zone": record.get("zone"),
            "floor": record.get("floor"),
            "episode_status": status,
            "bridge": record.get("bridge"),
            "completed": int(status == "won"),
            "deaths": int(status == "dead"),
            "raw_events": [dict(event) for event in record.get("events", [])],
            **record.get("metrics", {}),
        }
        return observation, info
