"""Hard qualification gate for the live NecroDancer controller path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from autodancer.constants import Action, GridChannel, PlayerFeature
from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.diagnose import LiveMechanicProbes
from autodancer.live.explore import LiveExplorer
from autodancer.live.native_pipe import NativePipeError
from autodancer.live.protocol import SCHEMA_VERSION, ProtocolError
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.training.async_collector import VersionedAsyncRolloutCollector
from autodancer.training.baseline import _evaluate_deterministic_async
from autodancer.training.model import ModelConfig, RecurrentActorCritic
from autodancer.training.train import resolve_device

PHASES = (
    "preflight",
    "conformance",
    "deterministic_replay",
    "forced_recovery",
    "collector_evaluation",
    "natural_soak",
)
REQUIRED_MECHANICS = {
    "move",
    "wall_attempt",
    "dig",
    "combat",
    "enemy_kill",
    "player_damage",
    "item_collected",
    "trap_seen",
    "death",
    "floor_transition",
    "zone_transition",
    "boss_entry",
}


class QualificationFailure(RuntimeError):
    """A controller acceptance criterion was not met."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wsp_entry_names(path: Path) -> set[str]:
    data = path.read_bytes()
    if not data.startswith(b"WSP16") or len(data) < 17:
        raise QualificationFailure("NecroDancer.wsp is not a WSP16 archive")
    names: set[str] = set()
    offset = 9
    while offset + 2 <= len(data):
        length = struct.unpack_from("<H", data, offset)[0]
        if not 0 < length < 512 or offset + 2 + length > len(data):
            break
        encoded = data[offset + 2 : offset + 2 + length]
        if any(byte < 32 or byte > 126 for byte in encoded):
            break
        names.add(encoded.decode("ascii"))
        offset += 2 + length
    return names


def _configuration(arguments: argparse.Namespace, count: int, phase: str) -> SupervisorConfig:
    return SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=count,
        startup_timeout=arguments.startup_timeout,
        turn_timeout=arguments.turn_timeout,
        reset_timeout=arguments.reset_timeout,
        max_turns=max(arguments.transitions_per_worker + 100, 10_000),
        affinity_policy=arguments.affinity,
        diagnostic_root=arguments.run_dir / "controller-diagnostics" / phase,
        qualification_mode=phase == "conformance",
    )


def preflight(arguments: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    built_dll = repository / "native" / "build" / "autodancer_native.dll"
    installed_dll = arguments.game_dir / "autodancer_native.dll"
    required = [
        arguments.game_dir / "Necrodancer.exe",
        arguments.game_dir / "NecroDancer.wsp",
        arguments.mod_dir / "mod.json",
        arguments.mod_dir / "scripts" / "Bridge.lua",
        arguments.mod_dir / "scripts" / "AutoDancer.lua",
        built_dll,
        installed_dll,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise QualificationFailure(f"Controller preflight is missing files: {missing}")
    bridge_source = (arguments.mod_dir / "scripts" / "Bridge.lua").read_text(
        encoding="utf-8"
    )
    telemetry_source = (arguments.mod_dir / "scripts" / "AutoDancer.lua").read_text(
        encoding="utf-8"
    )
    schema_literal = f"SCHEMA_VERSION = {SCHEMA_VERSION}"
    if schema_literal not in bridge_source or schema_literal not in telemetry_source:
        raise QualificationFailure("Lua and Python protocol schema versions do not match")
    built_hash = _sha256(built_dll)
    installed_hash = _sha256(installed_dll)
    if built_hash != installed_hash:
        raise QualificationFailure(
            "The installed native bridge does not match native/build/autodancer_native.dll; "
            "run native/build.ps1, then copy the built DLL into --game-dir"
        )
    native_module = "scripts/system/game/AutoDancerNative.lua"
    if native_module not in _wsp_entry_names(arguments.game_dir / "NecroDancer.wsp"):
        raise QualificationFailure(
            "NecroDancer.wsp does not contain the AutoDancer native loader; run "
            "tools/patch_wsp.py against the pinned game archive"
        )
    return {
        "passed": True,
        "schema_version": SCHEMA_VERSION,
        "native_sha256": built_hash,
        "wsp_native_module": native_module,
        "mod_files": {
            str(path.relative_to(arguments.mod_dir)): _sha256(path)
            for path in required[2:5]
        },
    }


def conformance(arguments: argparse.Namespace) -> dict[str, Any]:
    with AutoDancerSupervisor(_configuration(arguments, 1, "conformance")) as supervisor:
        environment = supervisor.environment("worker-0000")
        probes = LiveMechanicProbes(environment, step_delay=0.0)
        probes.verify_action_contract(47_000)
        probes.exercise_mechanics(list(range(47_001, 47_006)), 2_000)
        report = probes.ledger.report()
        _, initial_info = environment.reset(seed=47_999)
        boss_info = initial_info
        for target_level in (2, 3, 4):
            _, boss_info = environment.qualification_goto_level(target_level)
        boss_zone = int(boss_info.get("zone") or 0)
        boss_floor = int(boss_info.get("floor") or 0)
        if boss_floor < 4:
            raise QualificationFailure(
                f"Qualification level 4 did not load a boss floor: {boss_info}"
            )
        zone_observation, zone_info = environment.qualification_goto_level(5)
        next_zone = int(zone_info.get("zone") or 0)
        if next_zone <= boss_zone:
            raise QualificationFailure(
                f"Qualification level 5 did not cross a zone boundary: {zone_info}"
            )
        first_action_observation, _, terminated, truncated, first_action_info = (
            environment.step(Action.WAIT)
        )
        if terminated or truncated:
            raise QualificationFailure(
                "The first action after the qualification zone transition was terminal"
            )
        if not np.array_equal(
            zone_observation["player"][[PlayerFeature.ZONE, PlayerFeature.FLOOR]],
            first_action_observation["player"][
                [PlayerFeature.ZONE, PlayerFeature.FLOOR]
            ],
        ):
            raise QualificationFailure(
                "The first action after a level transition did not use the current level"
            )
        report["live_level_boundaries"] = {
            "passed": True,
            "initial": {
                "zone": initial_info.get("zone"),
                "floor": initial_info.get("floor"),
            },
            "boss_entry": {"zone": boss_zone, "floor": boss_floor},
            "zone_transition": {
                "zone": next_zone,
                "floor": zone_info.get("floor"),
            },
            "first_action": {
                "zone": first_action_info.get("zone"),
                "floor": first_action_info.get("floor"),
            },
        }
        environment.close()
    passed = bool(
        report["core_action_contract_passed"]
        and report["all_legal_actions_acknowledged"]
    )
    if not passed:
        raise QualificationFailure("Live action/mechanic conformance failed")
    return {"passed": True, **report}


def _normalized_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, original in observation.items():
        value = np.array(original, copy=True)
        if key == "player":
            value[
                [PlayerFeature.MUSIC_ELAPSED_DS, PlayerFeature.MUSIC_REMAINING_DS]
            ] = 0
        elif key == "grid":
            value[
                ...,
                [
                    GridChannel.TRAP_ACTIVATION_DS,
                    GridChannel.TRAP_FAILURE_DS,
                    GridChannel.TELL_ANIMATION_DS,
                ],
            ] = 0
        result[key] = value
    return result


def _normalized_info(info: dict[str, Any]) -> dict[str, Any]:
    events = [
        {key: value for key, value in event.items() if key != "entity_id"}
        for event in info.get("raw_events", [])
    ]
    return {
        "zone": info.get("zone"),
        "floor": info.get("floor"),
        "status": info.get("episode_status"),
        "events": events,
        "outcome": info.get("action_outcome"),
    }


def _normalized_signature(
    observation: dict[str, np.ndarray], info: dict[str, Any]
) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(_normalized_observation(observation).items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps(_normalized_info(info), sort_keys=True).encode())
    return digest.hexdigest()


def _replay_difference(
    observations: list[dict[str, np.ndarray]], infos: list[dict[str, Any]]
) -> dict[str, Any]:
    normalized = [_normalized_observation(value) for value in observations]
    details: dict[str, list[dict[str, Any]]] = {}
    for key in normalized[0]:
        indices = np.argwhere(normalized[0][key] != normalized[1][key])
        details[key] = [
            {
                "index": index.tolist(),
                "value_0": int(normalized[0][key][tuple(index)]),
                "value_1": int(normalized[1][key][tuple(index)]),
            }
            for index in indices[:20]
        ]
    return {
        "observation_differences": {
            key: int(np.count_nonzero(normalized[0][key] != normalized[1][key]))
            for key in normalized[0]
        },
        "difference_details": details,
        "info_0": _normalized_info(infos[0]),
        "info_1": _normalized_info(infos[1]),
    }


def deterministic_replay(arguments: argparse.Namespace) -> dict[str, Any]:
    with AutoDancerSupervisor(
        _configuration(arguments, 2, "deterministic-replay")
    ) as supervisor:
        environment = AutoDancerVectorEnv(supervisor)
        try:
            observation, infos = environment.reset([48_001, 48_001])
            matched = 0
            for step in range(512):
                signatures = [
                    _normalized_signature(
                        {key: value[index] for key, value in observation.items()}, infos[index]
                    )
                    for index in range(2)
                ]
                if signatures[0] != signatures[1]:
                    split = [
                        {key: value[index] for key, value in observation.items()}
                        for index in range(2)
                    ]
                    raise QualificationFailure(
                        "Deterministic replay diverged between worker slots at step "
                        f"{step}: {json.dumps(_replay_difference(split, infos), sort_keys=True)}"
                    )
                action = step % 5
                observation, _, terminated, truncated, infos = environment.step(
                    [action, action]
                )
                matched += 1
                if bool(np.any(terminated | truncated)):
                    if not bool(np.all(terminated | truncated)):
                        raise QualificationFailure(
                            "Same-seed workers terminated on different turns"
                        )
                    observation, infos = environment.reset([48_001 + step + 1] * 2)
            if environment.infrastructure_events:
                raise QualificationFailure("Replay encountered an infrastructure fault")
            return {"passed": True, "matched_transitions": matched}
        finally:
            environment.close()


def forced_recovery(arguments: argparse.Namespace) -> dict[str, Any]:
    with AutoDancerSupervisor(
        _configuration(arguments, arguments.num_instances, "forced-recovery")
    ) as supervisor:
        environment = AutoDancerVectorEnv(supervisor)
        try:
            environment.reset([49_000 + index for index in range(arguments.num_instances)])
            worker_id = environment.worker_ids[0]
            original = supervisor.workers[worker_id]
            process = supervisor._worker_processes[worker_id]
            process.terminate()
            process.wait(timeout=5)
            try:
                environment.environments[worker_id].step(int(Action.WAIT))
            except (TimeoutError, NativePipeError, ProtocolError) as error:
                failure = environment._failure(
                    0,
                    error,
                    operation="qualification_forced_termination",
                    context={"expected": True},
                )
                environment.recover(0, 49_999, failure=failure)
            else:
                raise QualificationFailure("Terminated worker unexpectedly acknowledged an action")
            replacement = supervisor.workers[worker_id]
            if replacement.pid == original.pid or replacement.restart_count != 1:
                raise QualificationFailure("Forced worker replacement identity is incorrect")
            if len(supervisor._worker_processes) != arguments.num_instances:
                raise QualificationFailure("Forced recovery did not restore exact capacity")
            return {
                "passed": True,
                "worker_id": worker_id,
                "old_pid": original.pid,
                "new_pid": replacement.pid,
                "restart_count": replacement.restart_count,
                "failure_bundle": replacement.failure_history[-1].get("bundle_path"),
            }
        finally:
            environment.close()


def collector_evaluation(arguments: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(arguments.device)
    with AutoDancerSupervisor(
        _configuration(arguments, arguments.num_instances, "collector-evaluation")
    ) as supervisor:
        environment = AutoDancerVectorEnv(supervisor)
        model = RecurrentActorCritic(
            ModelConfig(
                cell_size=16,
                spatial_size=32,
                hidden_size=16,
                entity_limit=8,
                attention_layers=1,
                attention_heads=4,
            )
        ).to(device)
        collector = VersionedAsyncRolloutCollector(
            environment,
            model,
            device=device,
            seed=50_001,
            action_contract="current",
        )
        try:
            rollout = collector.collect(16)
            episodes = _evaluate_deterministic_async(
                environment,
                model,
                seeds=[50_100 + index for index in range(arguments.num_instances)],
                max_steps=64,
                device=device,
                dashboard_state=None,
                action_contract="current",
            )
            if collector.last_runtime_metrics.get("collector_recoveries_total") != 0:
                raise QualificationFailure("Async collector required a natural recovery")
            if environment.infrastructure_events:
                raise QualificationFailure(
                    "Collector/evaluation integration had a controller fault"
                )
            return {
                "passed": True,
                "rollout_shape": list(rollout.actions.shape),
                "evaluation_episodes": len(episodes),
                "policy_version": collector.last_runtime_metrics["policy_version"],
            }
        finally:
            collector.close()
            environment.close()


def _memory_growth(samples: list[int]) -> float:
    if len(samples) < 20:
        return 0.0
    second_half = samples[len(samples) // 2 :]
    width = max(len(second_half) // 10, 1)
    beginning = float(np.mean(second_half[:width]))
    ending = float(np.mean(second_half[-width:]))
    return (ending - beginning) / max(beginning, 1.0)


def natural_soak(arguments: argparse.Namespace) -> dict[str, Any]:
    trace_root = arguments.run_dir / "soak-traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    with AutoDancerSupervisor(
        _configuration(arguments, arguments.num_instances, "natural-soak")
    ) as supervisor:
        environment = AutoDancerVectorEnv(supervisor)
        lock = threading.Lock()
        stop = threading.Event()
        global_mechanics: set[str] = set()
        progress_path = arguments.run_dir / "qualification-progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            boundaries = (
                progress.get("phases", {})
                .get("conformance", {})
                .get("live_level_boundaries", {})
            )
            if boundaries.get("passed"):
                global_mechanics.update({"boss_entry", "zone_transition"})

        def run_slot(index: int) -> dict[str, Any]:
            worker_id = environment.worker_ids[index]
            worker = environment.environments[worker_id]
            rng = np.random.default_rng(51_000 + index)
            seed = int(rng.integers(0, 2**31))
            reset_latencies: list[float] = []

            def reset_live(selected_seed: int, transition: int) -> tuple[dict, dict]:
                started = time.monotonic()
                try:
                    result = worker.reset(seed=selected_seed)
                except (TimeoutError, NativePipeError, ProtocolError) as error:
                    stop.set()
                    failure = environment._failure(
                        index,
                        error,
                        operation="qualification_natural_soak_reset",
                        context={"transition": transition, "seed": selected_seed},
                    )
                    supervisor.replace_worker(worker_id, failure=failure)
                    raise QualificationFailure(
                        f"Natural reset failure in {worker_id} at transition {transition}"
                    ) from error
                reset_latencies.append(time.monotonic() - started)
                return result

            observation, info = reset_live(seed, 0)
            explorer = LiveExplorer()
            latencies: list[float] = []
            memory: list[int] = []
            actions = [0] * 11
            mechanics: set[str] = set()
            previous_zone = int(info.get("zone") or 0)
            previous_floor = int(info.get("floor") or 0)
            maximum_zone = previous_zone
            maximum_floor = previous_floor
            trace_path = trace_root / f"{worker_id}.jsonl"
            trace_path.write_text("", encoding="utf-8")
            for transition in range(arguments.transitions_per_worker):
                if stop.is_set():
                    return {
                        "worker_id": worker_id,
                        "transitions": len(latencies),
                        "stopped_after_peer_failure": True,
                    }
                legal_specials = [
                    action
                    for action in range(5, 11)
                    if observation["action_mask"][action]
                ]
                if legal_specials and transition % 97 == index % 97:
                    action = min(legal_specials, key=lambda value: actions[value])
                else:
                    try:
                        action = int(explorer.choose(observation))
                    except RuntimeError:
                        action = int(Action.WAIT)
                started = time.monotonic()
                try:
                    next_observation, _, terminated, truncated, step_info = worker.step(action)
                except (TimeoutError, NativePipeError, ProtocolError) as error:
                    stop.set()
                    failure = environment._failure(
                        index,
                        error,
                        operation="qualification_natural_soak",
                        action=action,
                        context={
                            "transition": transition,
                            "seed": seed,
                            "run_id": info.get("run_id"),
                            "sequence": info.get("sequence"),
                        },
                    )
                    supervisor.replace_worker(worker_id, failure=failure)
                    raise QualificationFailure(
                        f"Natural controller failure in {worker_id} at transition {transition}"
                    ) from error
                latency = time.monotonic() - started
                latencies.append(latency)
                actions[action] += 1
                outcome = str((step_info.get("action_outcome") or {}).get("category", ""))
                if outcome:
                    mechanics.add(outcome)
                mechanics.update(
                    str(event.get("kind"))
                    for event in step_info.get("raw_events", [])
                )
                if np.any(next_observation["grid"][..., GridChannel.TRAP] > 0):
                    mechanics.add("trap_seen")
                zone = int(step_info.get("zone") or 0)
                floor = int(step_info.get("floor") or 0)
                maximum_zone = max(maximum_zone, zone)
                maximum_floor = max(maximum_floor, floor)
                if floor != previous_floor:
                    mechanics.add("floor_transition")
                    explorer.reset_level()
                if zone != previous_zone:
                    mechanics.add("zone_transition")
                if floor >= 4:
                    mechanics.add("boss_entry")
                previous_zone, previous_floor = zone, floor
                observation, info = next_observation, step_info
                if terminated or truncated:
                    if step_info.get("episode_status") == "dead":
                        mechanics.add("death")
                    seed = int(rng.integers(0, 2**31))
                    observation, info = reset_live(seed, transition + 1)
                    explorer.reset_level()
                    previous_zone = int(info.get("zone") or 0)
                    previous_floor = int(info.get("floor") or 0)
                if (transition + 1) % 1000 == 0:
                    handle = supervisor.workers[worker_id]
                    process = psutil.Process(handle.pid)
                    memory.append(int(process.memory_info().rss))
                    with trace_path.open("a", encoding="utf-8") as trace:
                        trace.write(
                            json.dumps(
                                {
                                    "transition": transition + 1,
                                    "seed": seed,
                                    "zone": previous_zone,
                                    "floor": previous_floor,
                                    "latency_p99_ms": float(
                                        np.percentile(latencies[-1000:], 99) * 1000
                                    ),
                                    "working_set_bytes": memory[-1],
                                    "max_frame_bytes": int(info.get("max_frame_bytes", 0)),
                                    "frame_bytes": int(info.get("frame_bytes", 0)),
                                    "outstanding_command_age_seconds": (
                                        supervisor.health()[worker_id].get(
                                            "outstanding_command_age_seconds"
                                        )
                                    ),
                                    "action_counts": list(actions),
                                    "mechanics": sorted(mechanics),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
            growth = _memory_growth(memory)
            result = {
                "worker_id": worker_id,
                "transitions": len(latencies),
                "action_counts": actions,
                "latency_p50_ms": float(np.percentile(latencies, 50) * 1000),
                "latency_p99_ms": float(np.percentile(latencies, 99) * 1000),
                "max_latency_seconds": max(latencies),
                "reset_latency_p50_ms": float(
                    np.percentile(reset_latencies, 50) * 1000
                ),
                "reset_latency_p99_ms": float(
                    np.percentile(reset_latencies, 99) * 1000
                ),
                "last_frame_bytes": int(info.get("frame_bytes", 0)),
                "max_frame_bytes": int(info.get("max_frame_bytes", 0)),
                "memory_growth_second_half": growth,
                "maximum_zone": maximum_zone,
                "maximum_floor": maximum_floor,
                "mechanics": sorted(mechanics),
            }
            with lock:
                global_mechanics.update(mechanics)
            return result

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=arguments.num_instances) as executor:
            futures = [executor.submit(run_slot, index) for index in range(arguments.num_instances)]
            workers = []
            try:
                for future in as_completed(futures):
                    workers.append(future.result())
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        workers.sort(key=lambda worker: worker["worker_id"])
        elapsed = time.monotonic() - started
        restarts = sum(handle.restart_count for handle in supervisor.workers.values())
        owned_identities = list(supervisor._owned_processes.items())
        criteria = {
            "exact_transitions": all(
                worker["transitions"] == arguments.transitions_per_worker
                for worker in workers
            ),
            "zero_natural_restarts": restarts == 0,
            "zero_infrastructure_events": not environment.infrastructure_events,
            "action_p99_below_250ms": all(
                worker["latency_p99_ms"] < 250 for worker in workers
            ),
            "no_unexplained_action_above_5s": all(
                worker["max_latency_seconds"] < 5 for worker in workers
            ),
            "stable_memory": all(
                worker["memory_growth_second_half"] <= 0.05 for worker in workers
            ),
            "mechanic_coverage": REQUIRED_MECHANICS <= global_mechanics,
            "exact_capacity": len(supervisor._worker_processes) == arguments.num_instances,
        }
        report = {
            "passed": False,
            "elapsed_seconds": elapsed,
            "transitions": sum(worker["transitions"] for worker in workers),
            "transitions_per_second": sum(worker["transitions"] for worker in workers)
            / max(elapsed, 1e-9),
            "worker_restarts": restarts,
            "mechanics": sorted(global_mechanics),
            "mechanic_sources": {
                "targeted_conformance": sorted(
                    global_mechanics & {"boss_entry", "zone_transition"}
                ),
                "natural_soak": sorted(
                    global_mechanics - {"boss_entry", "zone_transition"}
                ),
            },
            "missing_mechanics": sorted(REQUIRED_MECHANICS - global_mechanics),
            "criteria": criteria,
            "workers": workers,
        }
        environment.close()
    remaining_owned: list[int] = []
    for pid, create_time in owned_identities:
        try:
            process = psutil.Process(pid)
            if process.create_time() == create_time:
                remaining_owned.append(pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    report["remaining_owned_processes"] = remaining_owned
    report["criteria"]["complete_cleanup"] = not remaining_owned
    report["passed"] = all(report["criteria"].values())
    _atomic_json(arguments.run_dir / "natural-soak-report.json", report)
    if not report["passed"]:
        raise QualificationFailure(f"Natural soak failed: {report['criteria']}")
    return report


def _summary_markdown(report: dict[str, Any]) -> str:
    rows = ["# Live Controller Qualification", "", f"Passed: **{report['passed']}**", ""]
    for phase in PHASES:
        value = report["phases"].get(phase)
        if value is not None:
            rows.append(f"- {phase}: {'PASS' if value.get('passed') else 'FAIL'}")
    rows.extend(["", f"Generated: {report['completed_at']}"])
    return "\n".join(rows) + "\n"


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = arguments.run_dir / "qualification-progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if arguments.resume and progress_path.is_file()
        else {"schema_version": 1, "phases": {}}
    )
    functions = {
        "preflight": preflight,
        "conformance": conformance,
        "deterministic_replay": deterministic_replay,
        "forced_recovery": forced_recovery,
        "collector_evaluation": collector_evaluation,
        "natural_soak": natural_soak,
    }
    failure: str | None = None
    for phase in PHASES:
        if phase != "natural_soak" and progress["phases"].get(phase, {}).get("passed"):
            continue
        try:
            progress["phases"][phase] = functions[phase](arguments)
        except Exception as error:
            progress["phases"][phase] = {
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failure = f"{phase}: {type(error).__name__}: {error}"
            _atomic_json(progress_path, progress)
            break
        _atomic_json(progress_path, progress)
    report = {
        "schema_version": 1,
        "created_at": progress.get("created_at", datetime.now(UTC).isoformat()),
        "completed_at": datetime.now(UTC).isoformat(),
        "passed": failure is None and all(
            progress["phases"].get(phase, {}).get("passed", False) for phase in PHASES
        ),
        "failure": failure,
        "configuration": {
            "game_dir": str(arguments.game_dir),
            "mod_dir": str(arguments.mod_dir),
            "num_instances": arguments.num_instances,
            "transitions_per_worker": arguments.transitions_per_worker,
            "device": arguments.device,
        },
        "phases": progress["phases"],
    }
    _atomic_json(arguments.run_dir / "qualification.json", report)
    (arguments.run_dir / "SUMMARY.md").write_text(
        _summary_markdown(report), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualify the complete live controller path")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, required=True)
    parser.add_argument("--num-instances", type=int, default=8)
    parser.add_argument("--transitions-per-worker", type=int, default=125_000)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--turn-timeout", type=float, default=30.0)
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    parser.add_argument("--affinity", choices=("auto", "none", "spread"), default="none")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.num_instances != 8:
        parser.error("controller qualification requires exactly eight workers")
    if arguments.transitions_per_worker <= 0:
        parser.error("--transitions-per-worker must be positive")
    report = run(arguments)
    print(json.dumps({"passed": report["passed"], "failure": report["failure"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
