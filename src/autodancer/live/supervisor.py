"""Lifecycle management for native SYNCHRONY duplicate instances."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from autodancer.envs.live import AutoDancerLiveEnv
from autodancer.live.bridge import NativePipeCommandBridge
from autodancer.live.native_pipe import NativePipeServer, pipe_name
from autodancer.live.protocol import (
    SCHEMA_VERSION,
    SUPPORTED_GAME_VERSION,
    SUPPORTED_STEAM_BUILD,
    NativePipeTurnSource,
)
from autodancer.rewards import RewardConfig

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
    reset_timeout: float = 30.0
    max_turns: int = 10000
    profile_root: Path = Path(".runtime/live-profiles")
    reward_config: RewardConfig = RewardConfig()
    telemetry_transport: str = "native-pipe"
    worker_profile: str = "symbolic"
    affinity_policy: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_dir", Path(self.game_dir).resolve())
        object.__setattr__(self, "mod_dir", Path(self.mod_dir).resolve())
        object.__setattr__(self, "profile_root", Path(self.profile_root).resolve())
        if self.num_instances <= 0:
            raise ValueError("num_instances must be positive")
        if self.startup_timeout <= 0 or self.turn_timeout <= 0 or self.reset_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if self.telemetry_transport != "native-pipe":
            raise ValueError("live supervisors require telemetry_transport='native-pipe'")
        if self.worker_profile != "symbolic":
            raise ValueError("the supported worker profile is 'symbolic'")
        if self.affinity_policy not in {"auto", "none", "spread"}:
            raise ValueError("affinity_policy must be auto, none, or spread")

    @property
    def executable(self) -> Path:
        return self.game_dir / "Necrodancer.exe"


@dataclass(slots=True)
class InstanceHandle:
    instance_id: str
    log_path: Path
    pipe_name: str
    transport: NativePipeServer
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
    _pipe_servers: dict[str, NativePipeServer] = field(default_factory=dict, init=False)
    _worker_processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict, init=False)
    _owned_pids: set[int] = field(default_factory=set, init=False)

    @property
    def worker_ids(self) -> list[str]:
        return [f"worker-{index:04d}" for index in range(self.config.num_instances)]

    def start(self) -> AutoDancerSupervisor:
        self._validate_installation()
        self._refuse_existing_processes()
        self._prepare_pipes()
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
        native_library = self.config.game_dir / "autodancer_native.dll"
        if not native_library.is_file():
            raise SupervisorError(f"Native bridge library not found: {native_library}")
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

    def _prepare_pipes(self) -> None:
        for instance_id in self.worker_ids:
            name = pipe_name(self.session_id, instance_id)
            self._pipe_servers[instance_id] = NativePipeServer(name)

    def _launch_process(self, instance_id: str, arguments: list[str]) -> subprocess.Popen[bytes]:
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        environment = os.environ.copy()
        local_profile, roaming_profile = self._prepare_worker_profile(instance_id)
        environment["LOCALAPPDATA"] = str(local_profile)
        environment["APPDATA"] = str(roaming_profile)
        environment["AUTODANCER_INSTANCE_ID"] = instance_id
        environment["AUTODANCER_PIPE"] = self._pipe_servers[instance_id].name
        return subprocess.Popen(
            [str(self.config.executable), *arguments],
            cwd=self.config.game_dir,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )

    def _prepare_worker_profile(self, instance_id: str) -> tuple[Path, Path]:
        root = (self.config.profile_root / self.session_id / instance_id).resolve()
        expected = (self.config.profile_root / self.session_id).resolve()
        if expected not in root.parents:
            raise SupervisorError("worker profile escaped the owned session directory")
        local = root / "Local"
        roaming = root / "Roaming"
        destination_mod = local / "NecroDancer" / "mods" / "AutoDancer"
        shutil.copytree(self.config.mod_dir, destination_mod, dirs_exist_ok=True)
        source_roaming = Path(os.environ.get("APPDATA", "")) / "NecroDancer"
        destination_roaming = roaming / "NecroDancer"
        destination_roaming.mkdir(parents=True, exist_ok=True)
        for name in ("userconfig.json", "ConfigKeys.dat"):
            source = source_roaming / name
            if source.is_file():
                shutil.copy2(source, destination_roaming / name)
        user_config = destination_roaming / "userconfig.json"
        if user_config.is_file():
            try:
                payload = json.loads(user_config.read_text(encoding="utf-8"))
                game = payload.setdefault("wos", {}).setdefault("game", {})
                game.setdefault("window", {})["size"] = [320, 180]
                game["window"]["maximized"] = False
                user_config.write_text(json.dumps(payload), encoding="utf-8")
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        return local, roaming

    def _log_paths(self) -> list[Path]:
        return sorted(self.config.game_dir.glob("NecroDancer*.log"))

    def _log_offsets(self) -> dict[Path, tuple[int, int]]:
        return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in self._log_paths()}

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
        baseline: dict[Path, tuple[int, int]],
        *,
        role: str,
    ) -> Path:
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            for path in self._log_paths():
                previous = baseline.get(path)
                stat = path.stat()
                offset = (
                    previous[0] if previous is not None and previous[1] == stat.st_mtime_ns else 0
                )
                for record in self._ready_records(path, offset):
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
        baseline = self._log_offsets()
        config_name = f"AutoDancer-{worker_id}.lua"
        log_name = f"NecroDancer-{worker_id}.log"
        process = self._launch_process(
            worker_id,
            [
                f"-cwos.game.debug.logging.file.name={log_name}",
                "-cwos.game.debug.logging.file.flushInterval=0.05",
                "-cwos.game.debug.logging.console.verbosity=0",
                "-cwos.game.assets.autoReload.enabled=false",
                "-cwos.game.steam.enabled=false",
                "-cwos.game.galaxy.enabled=false",
                "-cwos.game.size=[320,180]",
            ],
        )
        self._worker_processes[worker_id] = process
        self._owned_pids.add(process.pid)
        self._apply_affinity(process.pid, self.worker_ids.index(worker_id))
        log_path = self._wait_for_ready(worker_id, baseline, role="worker")
        handle = InstanceHandle(
            instance_id=worker_id,
            log_path=log_path,
            pipe_name=self._pipe_servers[worker_id].name,
            transport=self._pipe_servers[worker_id],
            pid=process.pid,
            config_name=config_name,
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
            turn_source=NativePipeTurnSource(handle.transport),
            bridge=NativePipeCommandBridge(
                handle.transport,
                instance_id=worker_id,
                session_id=self.session_id,
                timeout=self.config.turn_timeout,
            ),
            turn_timeout=self.config.turn_timeout,
            reset_timeout=self.config.reset_timeout,
            max_turns=self.config.max_turns,
            instance_id=worker_id,
            reward_config=self.config.reward_config,
        )

    def replace_worker(self, worker_id: str) -> InstanceHandle:
        previous = self.workers[worker_id]
        previous.healthy = False
        process = self._worker_processes.pop(worker_id, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        previous.transport.close()
        name = pipe_name(self.session_id, worker_id)
        self._pipe_servers[worker_id] = NativePipeServer(name)
        return self._spawn_worker(worker_id, restart_count=previous.restart_count + 1)

    def health(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for worker_id, handle in self.workers.items():
            process_metrics: dict[str, Any] = {}
            if handle.pid is not None:
                try:
                    process = psutil.Process(handle.pid)
                    cpu = process.cpu_times()
                    process_metrics = {
                        "cpu_seconds": float(cpu.user + cpu.system),
                        "working_set_bytes": int(process.memory_info().rss),
                    }
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
            result[worker_id] = {
                "healthy": handle.healthy,
                "pid": handle.pid,
                "config_name": handle.config_name,
                "restart_count": handle.restart_count,
                "last_latency": handle.last_latency,
                "episode_status": handle.episode_status,
                "last_acknowledged_command": handle.last_acknowledged_command,
                **process_metrics,
            }
        return result

    def _apply_affinity(self, pid: int, slot: int) -> None:
        if self.config.affinity_policy == "none" or os.name != "nt":
            return
        logical = psutil.cpu_count(logical=True) or 1
        physical = psutil.cpu_count(logical=False) or logical
        if self.config.affinity_policy == "auto" and self.config.num_instances >= physical:
            return
        available = list(range(1, logical)) or [0]
        try:
            psutil.Process(pid).cpu_affinity([available[slot % len(available)]])
        except (AttributeError, psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            return

    def close(self) -> None:
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
        for server in self._pipe_servers.values():
            server.close()
        self._pipe_servers.clear()
        self._owned_pids.clear()
        session_profiles = (self.config.profile_root / self.session_id).resolve()
        profile_root = self.config.profile_root.resolve()
        if profile_root in session_profiles.parents and session_profiles.exists():
            shutil.rmtree(session_profiles)
        self._worker_processes.clear()

    def __enter__(self) -> AutoDancerSupervisor:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
