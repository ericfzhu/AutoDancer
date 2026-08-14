"""Gymnasium adapter for a focused live NecroDancer process."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from autodancer.constants import ACTION_COUNT, RGB_SIZE, Action
from autodancer.live.macos import (
    ActionSender,
    FrameCapture,
    MacOSActionSender,
    MacOSFrameCapture,
)
from autodancer.live.protocol import (
    JsonlTurnSource,
    TurnSource,
    decode_observation,
    validate_record,
)
from autodancer.observation import observation_space
from autodancer.rewards import reward_from_event_dicts


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
    ) -> None:
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        if turn_source is None and log_path is None:
            self._source: TurnSource | None = None
        else:
            self._source = turn_source or JsonlTurnSource(Path(log_path))  # type: ignore[arg-type]
        self._sender = action_sender
        self._capture = frame_capture
        self.render_mode = render_mode
        self.turn_timeout = turn_timeout
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = observation_space()
        self._last_observation: dict[str, np.ndarray] | None = None
        self._last_record: dict[str, Any] | None = None

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
        sender.restart()
        record = source.read(self.turn_timeout)
        validate_record(record)
        if record["kind"] != "reset":
            raise RuntimeError("The first live record after restart must have kind 'reset'")
        return self._accept_record(record)

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        source, sender = self._dependencies()
        selected = Action(int(action))
        if (
            self._last_observation is not None
            and not self._last_observation["action_mask"][selected]
        ):
            raise ValueError(f"Action {selected.name} is masked in the current live state")
        sender.send_action(selected)
        record = source.read(self.turn_timeout)
        validate_record(record)
        observation, info = self._accept_record(record)
        events = record.get("events", [])
        reward = reward_from_event_dicts(events)
        return (
            observation,
            reward,
            bool(record.get("terminated", False)),
            bool(record.get("truncated", False)),
            info,
        )

    def _accept_record(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation = decode_observation(record["observation"])
        self._last_observation = observation
        self._last_record = record
        info = {
            "sequence": record["sequence"],
            "seed": record.get("seed"),
            "character": record.get("character"),
            "game": record["game"],
            "zone": record.get("zone"),
            "floor": record.get("floor"),
            "raw_events": record.get("events", []),
            **record.get("metrics", {}),
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
