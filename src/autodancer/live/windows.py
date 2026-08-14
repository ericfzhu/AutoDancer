"""Windows input and capture implementations for live Bard runs."""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable, Mapping

import numpy as np

from autodancer.constants import RGB_SIZE, Action


DEFAULT_VIRTUAL_KEYS: dict[Action, int] = {
    Action.UP: 0x26,
    Action.RIGHT: 0x27,
    Action.DOWN: 0x28,
    Action.LEFT: 0x25,
    Action.WAIT: 0x20,
    Action.BOMB: 0x42,
    Action.ITEM_1: 0x31,
    Action.ITEM_2: 0x32,
    Action.THROW: 0x54,
    Action.SPELL_1: 0x34,
    Action.SPELL_2: 0x35,
}


class WindowsActionSender:
    """Send virtual-key presses to a focused NecroDancer window."""

    KEYEVENTF_KEYUP = 0x0002

    def __init__(
        self,
        virtual_keys: Mapping[Action, int] | None = None,
        *,
        restart_virtual_key: int = 0x77,
        require_focus: bool = True,
    ) -> None:
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("WindowsActionSender is only available on Windows")
        self._user32 = ctypes.windll.user32
        self.virtual_keys = dict(virtual_keys or DEFAULT_VIRTUAL_KEYS)
        self.restart_virtual_key = restart_virtual_key
        self.require_focus = require_focus

    def _assert_focus(self) -> None:
        if not self.require_focus:
            return
        window = self._user32.GetForegroundWindow()
        length = self._user32.GetWindowTextLengthW(window)
        title = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(window, title, length + 1)
        if "NecroDancer" not in title.value:
            raise RuntimeError(
                f"Keep Crypt of the NecroDancer focused; front window is {title.value!r}"
            )

    def _press(self, virtual_key: int) -> None:
        self._assert_focus()
        self._user32.keybd_event(virtual_key, 0, 0, 0)
        time.sleep(0.01)
        self._user32.keybd_event(virtual_key, 0, self.KEYEVENTF_KEYUP, 0)

    def send_action(self, action: Action) -> None:
        self._press(self.virtual_keys[action])

    def restart(self) -> None:
        self._press(self.restart_virtual_key)


class WindowsFrameCapture:
    def __init__(self, bounds: tuple[int, int, int, int] | None = None) -> None:
        try:
            from PIL import ImageGrab
        except ImportError as error:
            raise RuntimeError("Install AutoDancer with the 'windows' extra") from error
        self._grab: Callable[..., object] = ImageGrab.grab
        self.bounds = bounds

    def capture(self) -> np.ndarray:
        image = self._grab(bbox=self.bounds)
        image = image.convert("RGB").resize((RGB_SIZE, RGB_SIZE))  # type: ignore[attr-defined]
        return np.asarray(image, dtype=np.uint8)
