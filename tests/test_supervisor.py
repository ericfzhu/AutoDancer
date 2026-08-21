from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig, SupervisorError


def test_ready_log_discovery_preserves_worker_identity(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer-worker.log"
    marker = {
        "schema_version": 4,
        "instance_id": "worker-0007",
        "role": "worker",
        "game_version": "v4.2.1-b5713",
        "steam_build": "22938426",
        "command_file": "bridge-command.worker-0007.txt",
    }
    path.write_text("[Info] AUTODANCER_READY:" + json.dumps(marker) + "\n", encoding="utf-8")
    assert AutoDancerSupervisor._ready_records(path, 0) == [marker]


def test_supervisor_refuses_unowned_game_process(monkeypatch: pytest.MonkeyPatch) -> None:
    process = SimpleNamespace(info={"pid": 77, "name": "NecroDancer.exe", "exe": "game"})
    monkeypatch.setattr(
        "autodancer.live.supervisor.psutil.process_iter", lambda _: [process]
    )
    supervisor = AutoDancerSupervisor(
        SupervisorConfig(Path("game"), Path("mod"), num_instances=4)
    )
    with pytest.raises(SupervisorError, match=r"PIDs: \[77\]"):
        supervisor._refuse_existing_processes()


def test_supervisor_worker_slots_and_pipe_names_are_predictable(tmp_path: Path) -> None:
    config = SupervisorConfig(
        tmp_path / "game",
        tmp_path / "mod",
        num_instances=3,
        profile_root=tmp_path / "profiles",
    )
    supervisor = AutoDancerSupervisor(config)
    supervisor.session_id = "test-session"
    assert supervisor.worker_ids == ["worker-0000", "worker-0001", "worker-0002"]
    supervisor._prepare_pipes()
    try:
        assert [supervisor._pipe_servers[worker].name for worker in supervisor.worker_ids] == [
            rf"\\.\pipe\AutoDancer-test-session-worker-000{index}" for index in range(3)
        ]
    finally:
        for server in supervisor._pipe_servers.values():
            server.close()


def test_malformed_readiness_times_out_without_reducing_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SupervisorConfig(
        tmp_path / "game", tmp_path / "mod", num_instances=2, startup_timeout=0.01
    )
    supervisor = AutoDancerSupervisor(config)
    supervisor._coordinator = SimpleNamespace(poll=lambda: None)  # type: ignore[assignment]
    path = tmp_path / "NecroDancer.log"
    path.write_text(
        'AUTODANCER_READY:{"schema_version":3,"instance_id":"worker-0000"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(AutoDancerSupervisor, "_log_paths", lambda _: [path])
    with pytest.raises(SupervisorError, match="Timed out waiting for worker"):
        supervisor._wait_for_ready(
            "worker-0000", {path: (0, path.stat().st_mtime_ns)}, role="worker"
        )
