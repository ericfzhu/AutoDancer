from __future__ import annotations

import inspect

from autodancer.constants import Action
from autodancer.live.windows import DEFAULT_VIRTUAL_KEYS, VK_F8, WindowsActionSender


def test_windows_restart_defaults_to_f8() -> None:
    assert VK_F8 == 0x77
    assert inspect.signature(WindowsActionSender).parameters["restart_virtual_key"].default == VK_F8


def test_windows_maps_every_live_action_except_wait() -> None:
    assert set(DEFAULT_VIRTUAL_KEYS) == set(Action) - {Action.WAIT}
