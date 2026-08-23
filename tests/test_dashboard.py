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
        "inventory": np.zeros((1, INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16),
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
        assert "height: 100%; overflow: hidden" in html
        assert "function layoutCards(count)" in html
        assert 'class="stat-row ${className}"' in html
        assert 'statRow("Reward parts"' in html
        assert payload["status"] == "starting"
        assert payload["workers"] == []
    finally:
        server.stop()


def test_dashboard_updates_one_worker_without_replacing_others() -> None:
    state = DashboardState()
    observations = {
        "grid": np.zeros((2, GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=np.int16),
        "player": np.zeros((2, PLAYER_FEATURES), dtype=np.int32),
        "inventory": np.zeros((2, INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=np.int16),
        "action_mask": np.ones((2, ACTION_COUNT), dtype=np.int8),
    }
    for index in range(2):
        state.update_worker(
            index,
            f"worker-{index:04d}",
            {key: value[index] for key, value in observations.items()},
            {"turns": 0},
        )
    moved = {key: value[0].copy() for key, value in observations.items()}
    moved["player"][8] = 1
    state.update_worker(0, "worker-0000", moved, {"turns": 1}, action=3, reward=0.1)
    state.update_health({"worker-0000": {"healthy": True, "pid": 42}})
    snapshot = state.snapshot()
    assert [worker["instance_id"] for worker in snapshot["workers"]] == [
        "worker-0000",
        "worker-0001",
    ]
    assert snapshot["workers"][0]["info"]["turns"] == 1
    assert snapshot["workers"][0]["action"] == 3
    assert snapshot["workers"][0]["health"]["pid"] == 42
    assert snapshot["workers"][1]["info"]["turns"] == 0
