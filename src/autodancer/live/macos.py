"""macOS input and capture implementations for live Bard runs."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np

from autodancer.constants import RGB_SIZE, Action


class ActionSender(Protocol):
    def send_action(self, action: Action) -> None: ...

    def restart(self) -> None: ...


class FrameCapture(Protocol):
    def capture(self) -> np.ndarray: ...


DEFAULT_KEY_CODES: dict[Action, int] = {
    Action.UP: 126,
    Action.RIGHT: 124,
    Action.DOWN: 125,
    Action.LEFT: 123,
    Action.WAIT: 49,
    Action.BOMB: 11,
    Action.ITEM_1: 18,
    Action.ITEM_2: 19,
    Action.THROW: 17,
    Action.SPELL_1: 20,
    Action.SPELL_2: 21,
}


class MacOSActionSender:
    def __init__(
        self,
        key_codes: Mapping[Action, int] | None = None,
        *,
        restart_key_code: int = 100,
        require_focus: bool = True,
    ) -> None:
        try:
            import Quartz  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Install AutoDancer with the 'macos' extra") from error
        self._quartz = Quartz
        self.key_codes = dict(key_codes or DEFAULT_KEY_CODES)
        self.restart_key_code = restart_key_code
        self.require_focus = require_focus

    def _assert_focus(self) -> None:
        if not self.require_focus:
            return
        try:
            from AppKit import NSWorkspace  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("The macOS AppKit bridge is not installed") from error
        application = NSWorkspace.sharedWorkspace().frontmostApplication()
        name = str(application.localizedName() or "")
        if "NecroDancer" not in name:
            raise RuntimeError(f"Keep Crypt of the NecroDancer focused; front app is {name!r}")

    def _press(self, key_code: int) -> None:
        self._assert_focus()
        down = self._quartz.CGEventCreateKeyboardEvent(None, key_code, True)
        up = self._quartz.CGEventCreateKeyboardEvent(None, key_code, False)
        self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, down)
        time.sleep(0.01)
        self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, up)

    def send_action(self, action: Action) -> None:
        self._press(self.key_codes[action])

    def restart(self) -> None:
        self._press(self.restart_key_code)


class MacOSFrameCapture:
    def __init__(self, bounds: tuple[int, int, int, int] | None = None) -> None:
        try:
            from PIL import ImageGrab  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Install AutoDancer with the 'macos' extra") from error
        self._grab: Callable[..., object] = ImageGrab.grab
        self.bounds = bounds

    def capture(self) -> np.ndarray:
        image = self._grab(bbox=self.bounds)
        image = image.convert("RGB").resize((RGB_SIZE, RGB_SIZE))  # type: ignore[attr-defined]
        return np.asarray(image, dtype=np.uint8)

