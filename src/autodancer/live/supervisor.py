"""Lifecycle management for native SYNCHRONY duplicate instances."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from autodancer.envs.live import AutoDancerLiveEnv
from autodancer.live.bridge import CoordinatorBridge
from autodancer.live.protocol import (
    SCHEMA_VERSION,
    SUPPORTED_GAME_VERSION,
    SUPPORTED_STEAM_BUILD,
)

READY_MARKER = "AUTODANCER_READY:"


class SupervisorError(RuntimeError):
    """Raised when the requested fixed worker capacity cannot be maintained."""


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    game_dir: Path
    mod_dir: Path
    num_instances: int
    startup_timeout: float = 45.0
    turn_timeout: float = 10.0
    max_turns: int = 10000

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_dir", Path(self.game_dir).resolve())
        object.__setattr__(self, "mod_dir", Path(self.mod_dir).resolve())
        if self.num_instances <= 0:
            raise ValueError("num_instances must be positive")
        if self.startup_timeout <= 0 or self.turn_timeout <= 0:
            raise ValueError("timeouts must be positive")

    @property
    def executable(self) -> Path:
        return self.game_dir / "Necrodancer.exe"


@dataclass(slots=True)
class InstanceHandle:
    instance_id: str
    command_path: Path
    log_path: Path
    pid: int | None = None
    config_name: str = ""
    healthy: bool = True
    restart_count: int = 0
    last_latency: float = 0.0
    episode_status: str = "uninitialized"
    last_acknowledged_command: int = 0


@dataclass(slots=True)
class AutoDancerSupervisor:
    config: SupervisorConfig
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    workers: dict[str, InstanceHandle] = field(default_factory=dict, init=False)
    _coordinator: subprocess.Popen[bytes] | None = field(default=None, init=False)
    _coordinator_bridge: CoordinatorBridge | None = field(default=None, init=False)
    _owned_pids: set[int] = field(default_factory=set, init=False)

    @property
    def worker_ids(self) -> list[str]:
        return [f"worker-{index:04d}" for index in range(self.config.num_instances)]

    @property
    def coordinator_command_path(self) -> Path:
        return self.config.mod_dir / "bridge-command.coordinator.txt"

    def start(self) -> AutoDancerSupervisor:
        self._validate_installation()
        self._refuse_existing_processes()
        self._prepare_command_files()
        baseline = self._log_offsets()
        self._coordinator = self._launch_coordinator()
        self._owned_pids.add(self._coordinator.pid)
        self._wait_for_ready("coordinator", baseline, role="coordinator")
        self._coordinator_bridge = CoordinatorBridge(
            self.coordinator_command_path, session_id=self.session_id
        )
        try:
            for worker_id in self.worker_ids:
                self._spawn_worker(worker_id, restart_count=0)
        except Exception:
            self.close()
            raise
        return self

    def _validate_installation(self) -> None:
        if not self.config.executable.is_file():
            raise SupervisorError(f"Game executable not found: {self.config.executable}")
        for relative in ("mod.json", "scripts/AutoDancer.lua", "scripts/Bridge.lua"):
            if not (self.config.mod_dir / relative).is_file():
                raise SupervisorError(f"Installed mod is missing {relative}")

    def _refuse_existing_processes(self) -> None:
        existing: list[int] = []
        for process in psutil.process_iter(("pid", "name", "exe")):
            try:
                if str(process.info.get("name", "")).lower() == "necrodancer.exe":
                    existing.append(int(process.info["pid"]))
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        if existing:
            raise SupervisorError(
                "NecroDancer is already running; close unrelated instances first "
                f"(PIDs: {existing})"
            )

    def _prepare_command_files(self) -> None:
        paths = [self.coordinator_command_path]
        paths.extend(
            self.config.mod_dir / f"bridge-command.{worker_id}.txt"
            for worker_id in self.worker_ids
        )
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("NOOP\n", encoding="ascii")

    def _launch_coordinator(self) -> subprocess.Popen[bytes]:
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        return subprocess.Popen(
            [str(self.config.executable)],
            cwd=self.config.game_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )

    def _log_paths(self) -> list[Path]:
        return sorted(self.config.game_dir.glob("NecroDancer*.log"))

    def _log_offsets(self) -> dict[Path, int]:
        return {path: path.stat().st_size for path in self._log_paths()}

    @staticmethod
    def _ready_records(path: Path, offset: int) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        if path.stat().st_size < offset:
            offset = 0
        results: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            for line in handle:
                marker = line.find(READY_MARKER)
                if marker < 0:
                    continue
                payload = line[marker + len(READY_MARKER) :].lstrip()
                try:
                    record, _ = json.JSONDecoder().raw_decode(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    results.append(record)
        return results

    def _wait_for_ready(
        self,
        instance_id: str,
        baseline: dict[Path, int],
        *,
        role: str,
    ) -> Path:
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            if self._coordinator is not None and self._coordinator.poll() is not None:
                raise SupervisorError("Coordinator exited during startup")
            for path in self._log_paths():
                for record in self._ready_records(path, baseline.get(path, 0)):
                    if (
                        record.get("schema_version") == SCHEMA_VERSION
                        and record.get("instance_id") == instance_id
                        and record.get("role") == role
                        and record.get("game_version") == SUPPORTED_GAME_VERSION
                        and record.get("steam_build") == SUPPORTED_STEAM_BUILD
                    ):
                        return path
            time.sleep(0.05)
        raise SupervisorError(f"Timed out waiting for {role} {instance_id!r}")

    def _spawn_worker(self, worker_id: str, *, restart_count: int) -> InstanceHandle:
        if self._coordinator_bridge is None:
            raise SupervisorError("Coordinator bridge is not initialized")
        baseline = self._log_offsets()
        before_pids = self._game_pids()
        self._coordinator_bridge.spawn(worker_id)
        log_path = self._wait_for_ready(worker_id, baseline, role="worker")
        new_pids = self._game_pids() - before_pids
        self._owned_pids.update(new_pids)
        handle = InstanceHandle(
            instance_id=worker_id,
            command_path=self.config.mod_dir / f"bridge-command.{worker_id}.txt",
            log_path=log_path,
            pid=next(iter(new_pids), None),
            config_name=f"AutoDancer-{worker_id}.lua",
            restart_count=restart_count,
        )
        self.workers[worker_id] = handle
        return handle

    @staticmethod
    def _game_pids() -> set[int]:
        result: set[int] = set()
        for process in psutil.process_iter(("pid", "name")):
            try:
                if str(process.info.get("name", "")).lower() == "necrodancer.exe":
                    result.add(int(process.info["pid"]))
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        return result

    def environment(self, worker_id: str) -> AutoDancerLiveEnv:
        handle = self.workers[worker_id]
        return AutoDancerLiveEnv(
            log_path=handle.log_path,
            command_path=handle.command_path,
            turn_timeout=self.config.turn_timeout,
            max_turns=self.config.max_turns,
            instance_id=worker_id,
        )

    def replace_worker(self, worker_id: str) -> InstanceHandle:
        previous = self.workers[worker_id]
        previous.healthy = False
        if self._coordinator_bridge is None:
            raise SupervisorError("Coordinator bridge is unavailable")
        self._coordinator_bridge.close(worker_id)
        time.sleep(0.25)
        return self._spawn_worker(worker_id, restart_count=previous.restart_count + 1)

    def health(self) -> dict[str, dict[str, Any]]:
        return {
            worker_id: {
                "healthy": handle.healthy,
                "pid": handle.pid,
                "config_name": handle.config_name,
                "restart_count": handle.restart_count,
                "last_latency": handle.last_latency,
                "episode_status": handle.episode_status,
                "last_acknowledged_command": handle.last_acknowledged_command,
            }
            for worker_id, handle in self.workers.items()
        }

    def close(self) -> None:
        if self._coordinator_bridge is not None:
            for worker_id in list(self.workers):
                try:
                    self._coordinator_bridge.close(worker_id)
                except OSError:
                    pass
        time.sleep(0.1)
        for pid in sorted(self._owned_pids, reverse=True):
            try:
                process = psutil.Process(pid)
                process.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        owned_processes: list[psutil.Process] = []
        for pid in self._owned_pids:
            try:
                owned_processes.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned_processes, timeout=3)
        for process in alive:
            try:
                process.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
        self.workers.clear()
        self._owned_pids.clear()
        self._coordinator = None
        self._coordinator_bridge = None

    def __enter__(self) -> AutoDancerSupervisor:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
