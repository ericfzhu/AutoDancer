from __future__ import annotations

import json
from urllib.request import urlopen

import numpy as np

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    PLAYER_FEATURES,
)
from autodancer.training.dashboard import DashboardServer, DashboardState


def test_dashboard_state_serializes_symbolic_worker_telemetry() -> None:
    state = DashboardState()
    observation = {
        "grid": np.zeros((1, GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16),
        "player": np.zeros((1, PLAYER_FEATURES), dtype=np.int32),
        "inventory": np.zeros(
            (1, INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16
        ),
        "action_mask": np.ones((1, ACTION_COUNT), dtype=np.int8),
    }
    observation["grid"][0, 10, 10, 1] = 1
    state.set_status("training")
    state.update_training({"global_step": 64, "updates": 2})
    state.update_workers(
        ["worker-0000"],
        observation,
        [{"seed": 123, "episode_status": "running", "raw_events": []}],
        actions=np.asarray([3]),
        rewards=np.asarray([0.25]),
        health={"worker-0000": {"healthy": True, "pid": 99}},
    )
    snapshot = state.snapshot()
    assert snapshot["status"] == "training"
    assert snapshot["training"]["global_step"] == 64
    assert snapshot["workers"][0]["grid"][10][10][1] == 1
    assert snapshot["workers"][0]["action"] == 3
    assert snapshot["workers"][0]["reward"] == 0.25
    assert snapshot["workers"][0]["health"]["pid"] == 99


def test_dashboard_server_serves_html_and_json() -> None:
    state = DashboardState()
    server = DashboardServer(state, port=0).start()
    try:
        with urlopen(server.url, timeout=2) as response:  # noqa: S310
            html = response.read().decode()
        with urlopen(f"{server.url}api/state", timeout=2) as response:  # noqa: S310
            payload = json.load(response)
        assert "AutoDancer — live Bard workers" in html
        assert payload["status"] == "starting"
        assert payload["workers"] == []
    finally:
        server.stop()
