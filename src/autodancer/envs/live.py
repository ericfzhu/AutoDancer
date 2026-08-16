"""Gymnasium adapter for a focused live NecroDancer process."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from autodancer.constants import (
    ACTION_COUNT,
    GRID_SIZE,
    RGB_SIZE,
    Action,
    ActorKind,
    GridChannel,
    PlayerFeature,
    Terrain,
)
from autodancer.live.macos import (
    ActionSender,
    FrameCapture,
    MacOSActionSender,
    MacOSFrameCapture,
)
from autodancer.live.protocol import (
    JsonlTurnSource,
    ProtocolError,
    TurnSource,
    decode_observation,
    validate_record,
)
from autodancer.observation import observation_space
from autodancer.rewards import reward_from_event_dicts
from autodancer.tasks import TASKS


class AutoDancerLiveEnv(gym.Env[dict[str, np.ndarray], int]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        *,
        log_path: str | Path | None = None,
        turn_source: TurnSource | None = None,
        action_sender: ActionSender | None = None,
        frame_capture: FrameCapture | None = None,
        render_mode: str | None = None,
        turn_timeout: float = 5.0,
        attach_existing: bool = False,
        task: str = "all_zones",
        max_turns: int | None = None,
    ) -> None:
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}. Choose one of {sorted(TASKS)}")
        if turn_source is None and log_path is None:
            self._source: TurnSource | None = None
        else:
            self._source = turn_source or JsonlTurnSource(Path(log_path))  # type: ignore[arg-type]
        self._sender = action_sender
        self._capture = frame_capture
        self.render_mode = render_mode
        self.turn_timeout = float(turn_timeout)
        self.attach_existing = bool(attach_existing)
        self.task_name = task
        self.task_index = tuple(TASKS).index(task)
        self.max_turns = TASKS[task].max_turns if max_turns is None else int(max_turns)
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = observation_space()
        self._last_observation: dict[str, np.ndarray] | None = None
        self._last_record: dict[str, Any] | None = None
        self._episode_steps = 0
        self._episode_done = False

    def _dependencies(self) -> tuple[TurnSource, ActionSender]:
        if self._source is None:
            raise RuntimeError("Set log_path or inject turn_source before calling reset()")
        if self._sender is None:
            if sys.platform == "darwin":
                self._sender = MacOSActionSender()
            elif sys.platform == "win32":
                from autodancer.live.windows import WindowsActionSender

                self._sender = WindowsActionSender()
            else:
                raise RuntimeError(f"Live action sending is unsupported on {sys.platform!r}")
        return self._source, self._sender

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        source, sender = self._dependencies()
        source.reset_sequence()
        if self.attach_existing:
            record = source.read_latest(self.turn_timeout)
        else:
            sender.restart()
            record = source.read(self.turn_timeout)
        validate_record(record)
        if not self.attach_existing and record["kind"] != "reset":
            raise ProtocolError("The first live record after restart must have kind 'reset'")
        self._episode_steps = 0
        self._episode_done = record["episode_status"] != "running"
        return self._accept_record(record)

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("step() was called after the live episode ended; call reset()")
        source, sender = self._dependencies()
        try:
            selected = Action(int(action))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Action must be an integer from 0 to {ACTION_COUNT - 1}") from error
        if (
            self._last_observation is not None
            and not self._last_observation["action_mask"][selected]
        ):
            raise ValueError(f"Action {selected.name} is masked in the current live state")

        sender.send_action(selected)
        record = source.read(self.turn_timeout)
        validate_record(record)
        if record["kind"] == "reset":
            raise ProtocolError("A new/reset run record arrived while waiting for an action result")

        observation, info = self._accept_record(record)
        self._episode_steps += 1
        events = [dict(event) for event in record.get("events", [])]
        status = str(record["episode_status"])
        terminated = status in {"won", "dead"}
        truncated = status == "aborted"

        if not terminated and not truncated and self._episode_steps >= self.max_turns:
            truncated = True
            status = "aborted"
            events.append(
                {
                    "kind": "failure",
                    "amount": 1,
                    "data": {"reason": "client_turn_limit", "max_turns": self.max_turns},
                }
            )
            info["episode_status"] = status
            info["client_turn_limit"] = self.max_turns

        info["raw_events"] = events
        info["completed"] = int(status == "won")
        info["deaths"] = int(status == "dead")
        info["turns"] = self._episode_steps
        reward = reward_from_event_dicts(events)
        self._episode_done = terminated or truncated
        return observation, reward, terminated, truncated, info

    def _adapt_observation(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        observation = decode_observation(record["observation"])
        grid = observation["grid"]
        player = observation["player"]
        visible_enemies = int(
            np.count_nonzero(
                (grid[..., GridChannel.ACTOR] > ActorKind.PLAYER)
                & (grid[..., GridChannel.VISIBILITY] == 2)
            )
        )
        centre = GRID_SIZE // 2
        status = str(record["episode_status"])
        player[PlayerFeature.VISIBLE_ENEMIES] = visible_enemies
        player[PlayerFeature.ON_STAIRS] = int(
            grid[centre, centre, GridChannel.TERRAIN] == Terrain.STAIRS
        )
        player[PlayerFeature.TASK] = self.task_index
        player[PlayerFeature.WON] = int(status == "won")
        player[PlayerFeature.DEAD] = int(status == "dead")
        return observation

    def _accept_record(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation = self._adapt_observation(record)
        self._last_observation = observation
        self._last_record = record
        status = str(record["episode_status"])
        metrics = record.get("metrics", {})
        info = {
            "protocol_schema_version": record["schema_version"],
            "run_id": record["run_id"],
            "sequence": record["sequence"],
            "seed": record.get("seed"),
            "character": record.get("character"),
            "game": record["game"],
            "zone": record.get("zone"),
            "floor": record.get("floor"),
            "episode_status": status,
            "completed": int(status == "won"),
            "deaths": int(status == "dead"),
            "raw_events": [dict(event) for event in record.get("events", [])],
            **metrics,
        }
        return observation, info

    def render(self) -> np.ndarray:
        if self._capture is None:
            if sys.platform == "darwin":
                self._capture = MacOSFrameCapture()
            elif sys.platform == "win32":
                from autodancer.live.windows import WindowsFrameCapture

                self._capture = WindowsFrameCapture()
            else:
                raise RuntimeError(f"Live frame capture is unsupported on {sys.platform!r}")
        image = self._capture.capture()
        if image.shape != (RGB_SIZE, RGB_SIZE, 3):
            raise RuntimeError(
                f"Live capture has shape {image.shape}; expected {(RGB_SIZE, RGB_SIZE, 3)}"
            )
        return image
