from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodancer.live.native_pipe import pipe_name
from autodancer.live.protocol import SCHEMA_VERSION
from autodancer.live.supervisor import (
    AutoDancerSupervisor,
    InstanceHandle,
    SupervisorConfig,
    SupervisorError,
    synchronize_authoritative_mod,
)


def test_ready_log_discovery_preserves_worker_identity(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer-worker.log"
    marker = {
        "message_type": "hello",
        "schema_version": SCHEMA_VERSION,
        "instance_id": "worker-0007",
        "role": "worker",
        "session_id": "test-session",
        "launch_id": "test-launch",
        "game_version": "v4.2.1-b5713",
        "steam_build": "22938426",
        "command_file": "bridge-command.worker-0007.txt",
    }
    path.write_text("[Info] AUTODANCER_READY:" + json.dumps(marker) + "\n", encoding="utf-8")
    assert AutoDancerSupervisor._ready_records(path, 0) == [marker]


def test_supervisor_refuses_unowned_game_process(monkeypatch: pytest.MonkeyPatch) -> None:
    process = SimpleNamespace(info={"pid": 77, "name": "NecroDancer.exe", "exe": "game"})
    monkeypatch.setattr("autodancer.live.supervisor.psutil.process_iter", lambda _: [process])
    supervisor = AutoDancerSupervisor(SupervisorConfig(Path("game"), Path("mod"), num_instances=4))
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
    assert [pipe_name(supervisor.session_id, worker) for worker in supervisor.worker_ids] == [
        rf"\\.\pipe\AutoDancer-test-session-worker-000{index}" for index in range(3)
    ]


def test_qualification_startup_fault_slot_must_be_inside_capacity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outside worker capacity"):
        SupervisorConfig(
            tmp_path / "game",
            tmp_path / "mod",
            num_instances=2,
            qualification_startup_fault_slot=2,
        )


def test_curriculum_target_must_follow_start_level(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="after curriculum_start_level"):
        SupervisorConfig(
            tmp_path / "game",
            tmp_path / "mod",
            num_instances=1,
            curriculum_start_level=4,
            curriculum_target_level=4,
        )


def test_start_recovers_a_worker_that_fails_after_handle_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = AutoDancerSupervisor(
        SupervisorConfig(tmp_path / "game", tmp_path / "mod", num_instances=2)
    )
    monkeypatch.setattr(AutoDancerSupervisor, "_validate_installation", lambda _self: None)
    monkeypatch.setattr(AutoDancerSupervisor, "_refuse_existing_processes", lambda _self: None)
    attempts: list[tuple[str, int]] = []

    def spawn(
        _self: AutoDancerSupervisor, worker_id: str, restart_count: int
    ) -> InstanceHandle:
        attempts.append((worker_id, restart_count))
        handle = SimpleNamespace(instance_id=worker_id, restart_count=restart_count)
        supervisor.workers[worker_id] = handle
        if worker_id == "worker-0000" and restart_count == 0:
            raise SupervisorError("injected startup failure")
        return handle  # type: ignore[return-value]

    def replace(
        _self: AutoDancerSupervisor,
        worker_id: str,
        *,
        failure: dict[str, object],
    ) -> InstanceHandle:
        assert failure["operation"] == "worker_startup"
        return spawn(supervisor, worker_id, 1)

    monkeypatch.setattr(AutoDancerSupervisor, "_spawn_worker", spawn)
    monkeypatch.setattr(AutoDancerSupervisor, "replace_worker", replace)

    assert supervisor.start() is supervisor
    assert attempts == [
        ("worker-0000", 0),
        ("worker-0000", 1),
        ("worker-0001", 0),
    ]


def test_malformed_or_stale_pipe_hello_is_rejected(tmp_path: Path) -> None:
    config = SupervisorConfig(
        tmp_path / "game", tmp_path / "mod", num_instances=2, startup_timeout=0.01
    )
    supervisor = AutoDancerSupervisor(config)
    stale = {
        "message_type": "hello",
        "schema_version": SCHEMA_VERSION,
        "instance_id": "worker-0000",
        "role": "worker",
        "session_id": "wrong-session",
        "launch_id": "old-launch",
        "game_version": "v4.2.1-b5713",
        "steam_build": "22938426",
        "engine_state": {},
    }
    transport = SimpleNamespace(receive=lambda _timeout: json.dumps(stale).encode())
    with pytest.raises(SupervisorError, match="rejected HELLO"):
        supervisor._wait_for_hello("worker-0000", "new-launch", transport)  # type: ignore[arg-type]


def test_authoritative_mod_is_atomically_deployed_and_old_copy_preserved(
    tmp_path: Path,
) -> None:
    source = tmp_path / "repository-mod"
    destination = tmp_path / "game-mods" / "AutoDancer"
    backup = tmp_path / "diagnostics" / "backups"
    (source / "scripts").mkdir(parents=True)
    (source / "mod.json").write_text("new", encoding="utf-8")
    (source / "scripts" / "Bridge.lua").write_text("schema 10", encoding="utf-8")
    destination.mkdir(parents=True)
    (destination / "mod.json").write_text("old", encoding="utf-8")

    assert synchronize_authoritative_mod(source, destination, backup_root=backup)
    assert (destination / "mod.json").read_text(encoding="utf-8") == "new"
    assert (destination / "scripts" / "Bridge.lua").is_file()
    preserved = list(backup.glob("AutoDancer-*/mod.json"))
    assert len(preserved) == 1
    assert preserved[0].read_text(encoding="utf-8") == "old"
    assert not synchronize_authoritative_mod(source, destination, backup_root=backup)
