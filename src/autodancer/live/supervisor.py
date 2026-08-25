"""Lifecycle management for native SYNCHRONY duplicate instances."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from autodancer.envs.live import AutoDancerLiveEnv
from autodancer.live.bridge import NativePipeCommandBridge
from autodancer.live.native_pipe import NativePipeServer, pipe_name
from autodancer.live.protocol import NativePipeTurnSource, decode_pipe_message, validate_hello
from autodancer.rewards import RewardConfig

READY_MARKER = "AUTODANCER_READY:"


class SupervisorError(RuntimeError):
    """Raised when the requested fixed worker capacity cannot be maintained."""


def _directory_manifest(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def synchronize_authoritative_mod(
    source: Path, destination: Path, *, backup_root: Path
) -> bool:
    """Atomically install the repository mod when the game's shared mod is stale."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return False
    if destination.is_dir() and _directory_manifest(source) == _directory_manifest(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".AutoDancer-schema10-", dir=destination.parent))
    backup: Path | None = None
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        if _directory_manifest(source) != _directory_manifest(temporary):
            raise SupervisorError("The staged AutoDancer mod copy failed hash verification")
        if destination.exists():
            backup_root.mkdir(parents=True, exist_ok=True)
            backup = backup_root / f"AutoDancer-{int(time.time() * 1000)}"
            shutil.move(str(destination), str(backup))
        try:
            os.replace(temporary, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                shutil.move(str(backup), str(destination))
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return True


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
    diagnostic_root: Path = Path(".runtime/live-diagnostics")
    reward_config: RewardConfig = RewardConfig()
    telemetry_transport: str = "native-pipe"
    worker_profile: str = "symbolic"
    affinity_policy: str = "auto"
    qualification_mode: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_dir", Path(self.game_dir).resolve())
        object.__setattr__(self, "mod_dir", Path(self.mod_dir).resolve())
        object.__setattr__(self, "profile_root", Path(self.profile_root).resolve())
        object.__setattr__(self, "diagnostic_root", Path(self.diagnostic_root).resolve())
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
    launch_id: str = ""
    attempt: int = 0
    process_create_time: float | None = None
    pid: int | None = None
    config_name: str = ""
    healthy: bool = True
    restart_count: int = 0
    last_latency: float = 0.0
    episode_status: str = "uninitialized"
    last_acknowledged_command: int = 0
    command_lifecycle: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=128)
    )
    last_heartbeat: dict[str, Any] | None = None
    outstanding_command_since: float | None = None
    last_frame_bytes: int = 0
    max_frame_bytes: int = 0
    failure_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class AutoDancerSupervisor:
    config: SupervisorConfig
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    workers: dict[str, InstanceHandle] = field(default_factory=dict, init=False)
    _pipe_servers: dict[str, NativePipeServer] = field(default_factory=dict, init=False)
    _worker_processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict, init=False)
    _owned_processes: dict[int, float] = field(default_factory=dict, init=False)

    @property
    def worker_ids(self) -> list[str]:
        return [f"worker-{index:04d}" for index in range(self.config.num_instances)]

    def start(self) -> AutoDancerSupervisor:
        self._validate_installation()
        self._refuse_existing_processes()
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
        local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
        if not local_app_data.is_dir():
            raise SupervisorError("LOCALAPPDATA is unavailable for game mod deployment")
        synchronize_authoritative_mod(
            self.config.mod_dir,
            local_app_data / "NecroDancer" / "mods" / "AutoDancer",
            backup_root=self.config.diagnostic_root
            / self.session_id
            / "installed-mod-backups",
        )

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

    def _launch_process(
        self,
        instance_id: str,
        launch_id: str,
        transport: NativePipeServer,
        arguments: list[str],
    ) -> subprocess.Popen[bytes]:
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
        environment["AUTODANCER_LAUNCH_ID"] = launch_id
        environment["AUTODANCER_SUPERVISOR_SESSION"] = self.session_id
        environment["AUTODANCER_PIPE"] = transport.name
        environment["AUTODANCER_QUALIFICATION"] = (
            "1" if self.config.qualification_mode else "0"
        )
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

    def _wait_for_hello(
        self, instance_id: str, launch_id: str, transport: NativePipeServer
    ) -> None:
        try:
            message = decode_pipe_message(transport.receive(self.config.startup_timeout))
            validate_hello(
                message,
                instance_id=instance_id,
                session_id=self.session_id,
                launch_id=launch_id,
            )
        except Exception as error:
            raise SupervisorError(
                f"Timed out or rejected HELLO for worker {instance_id!r} "
                f"launch {launch_id!r}: {error}"
            ) from error

    def _spawn_worker(self, worker_id: str, *, restart_count: int) -> InstanceHandle:
        attempt = restart_count
        launch_id = f"{worker_id}-a{attempt:04d}-{uuid.uuid4().hex[:12]}"
        name = pipe_name(self.session_id, launch_id)
        transport = NativePipeServer(name)
        self._pipe_servers[worker_id] = transport
        config_name = f"AutoDancer-{worker_id}.lua"
        log_name = f"NecroDancer-{self.session_id[:12]}-{worker_id}-a{attempt:04d}.log"
        log_path = self.config.game_dir / log_name
        process = self._launch_process(
            worker_id,
            launch_id,
            transport,
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
        create_time = psutil.Process(process.pid).create_time()
        self._owned_processes[process.pid] = create_time
        self._apply_affinity(process.pid, self.worker_ids.index(worker_id))
        handle = InstanceHandle(
            instance_id=worker_id,
            log_path=log_path,
            pipe_name=self._pipe_servers[worker_id].name,
            transport=transport,
            launch_id=launch_id,
            attempt=attempt,
            process_create_time=create_time,
            pid=process.pid,
            config_name=config_name,
            restart_count=restart_count,
        )
        self.workers[worker_id] = handle
        self._wait_for_hello(worker_id, launch_id, transport)
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

        def record_status(status: dict[str, Any]) -> None:
            handle.command_lifecycle.append(status)
            phase = status.get("phase")
            if phase == "received":
                handle.outstanding_command_since = time.monotonic()
            elif phase in {"telemetry_sent", "command_error"}:
                handle.outstanding_command_since = None
            if status.get("phase") == "heartbeat":
                handle.last_heartbeat = status

        def record_progress(info: dict[str, Any]) -> None:
            bridge = info.get("bridge") or {}
            handle.last_acknowledged_command = int(bridge.get("command_id") or 0)
            handle.episode_status = str(info.get("episode_status") or "unknown")
            handle.last_latency = float(
                info.get("action_latency_seconds", info.get("reset_latency_seconds", 0.0))
            )
            handle.max_frame_bytes = max(
                handle.max_frame_bytes, int(info.get("max_frame_bytes", 0))
            )
            handle.last_frame_bytes = int(info.get("frame_bytes", 0))

        source = NativePipeTurnSource(
            handle.transport,
            instance_id=worker_id,
            session_id=self.session_id,
            launch_id=handle.launch_id,
            status_callback=record_status,
        )
        return AutoDancerLiveEnv(
            turn_source=source,
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
            session_id=self.session_id,
            launch_id=handle.launch_id,
            progress_callback=record_progress,
        )

    def replace_worker(
        self, worker_id: str, *, failure: dict[str, Any] | None = None
    ) -> InstanceHandle:
        previous = self.workers[worker_id]
        previous.healthy = False
        if failure is not None:
            bundle = self._capture_failure(previous, failure)
            previous.failure_history.append(bundle)
        process = self._worker_processes.pop(worker_id, None)
        if process is not None:
            self._terminate_process(process)
            self._owned_processes.pop(process.pid, None)
        self._archive_log(previous)
        previous.transport.close()
        last_error: Exception | None = None
        for retry in range(3):
            try:
                replacement = self._spawn_worker(
                    worker_id, restart_count=previous.restart_count + 1 + retry
                )
                replacement.failure_history = previous.failure_history
                return replacement
            except Exception as error:
                last_error = error
                failed_handle = self.workers.get(worker_id)
                if failed_handle is not None and failed_handle is not previous:
                    self._archive_log(failed_handle)
                failed = self._worker_processes.pop(worker_id, None)
                if failed is not None:
                    self._terminate_process(failed)
                    self._owned_processes.pop(failed.pid, None)
                server = self._pipe_servers.pop(worker_id, None)
                if server is not None:
                    server.close()
        raise SupervisorError(f"Could not restore fixed-capacity slot {worker_id}") from last_error

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def _archive_log(self, handle: InstanceHandle) -> Path | None:
        if not handle.log_path.is_file():
            return None
        destination = self.config.diagnostic_root / self.session_id / "worker-logs"
        destination.mkdir(parents=True, exist_ok=True)
        archived = destination / handle.log_path.name
        shutil.copy2(handle.log_path, archived)
        return archived

    def _capture_failure(
        self, handle: InstanceHandle, failure: dict[str, Any]
    ) -> dict[str, Any]:
        process = self._worker_processes.get(handle.instance_id)
        process_state: dict[str, Any] = {
            "pid": handle.pid,
            "create_time": handle.process_create_time,
            "alive": bool(process is not None and process.poll() is None),
            "exit_code": None if process is None else process.poll(),
        }
        if handle.pid is not None:
            try:
                observed = psutil.Process(handle.pid)
                if (
                    handle.process_create_time is not None
                    and observed.create_time() == handle.process_create_time
                ):
                    cpu = observed.cpu_times()
                    process_state.update(
                        cpu_seconds=float(cpu.user + cpu.system),
                        working_set_bytes=int(observed.memory_info().rss),
                    )
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
        log_tail: list[str] = []
        if handle.log_path.is_file():
            log_tail = handle.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-200:]
        payload = {
            "schema_version": 1,
            "captured_at": time.time(),
            "instance_id": handle.instance_id,
            "session_id": self.session_id,
            "launch_id": handle.launch_id,
            "attempt": handle.attempt,
            "process": process_state,
            "last_acknowledged_command": handle.last_acknowledged_command,
            "last_heartbeat": handle.last_heartbeat,
            "outstanding_command_age_seconds": (
                None
                if handle.outstanding_command_since is None
                else time.monotonic() - handle.outstanding_command_since
            ),
            "last_frame_bytes": handle.last_frame_bytes,
            "max_frame_bytes": handle.max_frame_bytes,
            "command_lifecycle": list(handle.command_lifecycle),
            "failure": failure,
            "log_path": str(handle.log_path),
            "log_tail": log_tail,
        }
        destination = self.config.diagnostic_root / self.session_id / "failures"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / (
            f"{handle.instance_id}-{handle.launch_id}-{int(time.time() * 1000)}.json"
        )
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        payload["bundle_path"] = str(path)
        return payload

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
                "launch_id": handle.launch_id,
                "attempt": handle.attempt,
                "config_name": handle.config_name,
                "restart_count": handle.restart_count,
                "last_latency": handle.last_latency,
                "episode_status": handle.episode_status,
                "last_acknowledged_command": handle.last_acknowledged_command,
                "last_heartbeat": handle.last_heartbeat,
                "outstanding_command_age_seconds": (
                    None
                    if handle.outstanding_command_since is None
                    else time.monotonic() - handle.outstanding_command_since
                ),
                "last_frame_bytes": handle.last_frame_bytes,
                "max_frame_bytes": handle.max_frame_bytes,
                "failure_count": len(handle.failure_history),
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
        for worker_id, process in list(self._worker_processes.items()):
            self._terminate_process(process)
            self._owned_processes.pop(process.pid, None)
            handle = self.workers.get(worker_id)
            if handle is not None:
                self._archive_log(handle)
        self.workers.clear()
        for server in self._pipe_servers.values():
            server.close()
        self._pipe_servers.clear()
        self._owned_processes.clear()
        session_profiles = (self.config.profile_root / self.session_id).resolve()
        profile_root = self.config.profile_root.resolve()
        if profile_root in session_profiles.parents and session_profiles.exists():
            shutil.rmtree(session_profiles)
        self._worker_processes.clear()

    def __enter__(self) -> AutoDancerSupervisor:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
