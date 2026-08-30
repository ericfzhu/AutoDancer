"""Build and qualify successful action traces against fresh live game resets."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from autodancer.constants import ACTION_COUNT, PlayerFeature
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.progress import deeper_level
from autodancer.training.demonstrations import (
    build_demonstration_bank,
    iter_trace_actions,
    normalized_observation_digest,
    validate_demonstration_bank,
    validate_demonstration_sources,
    write_demonstration_bank,
)
from autodancer.training.train import default_mod_dir


class ReplayEnvironment(Protocol):
    def reset(
        self, *, seed: int, options: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]: ...


def _event_counts(events: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in events or []:
        if not isinstance(event, Mapping):
            continue
        kind = str(event.get("kind") or event.get("type") or "")
        if kind:
            counts[kind] += 1
    return counts


def replay_trace(environment: ReplayEnvironment, trace: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one trace and compare its terminal gameplay evidence exactly."""

    started = time.monotonic()
    seed = int(trace["seed"])
    reset = dict(trace["curriculum_reset"])
    observation, reset_info = environment.reset(seed=seed, options={"curriculum": reset})
    turn_digests = [normalized_observation_digest(observation)]
    actual_events: Counter[str] = Counter()
    furthest = (
        int(reset_info.get("zone") or observation["player"][PlayerFeature.ZONE]),
        int(reset_info.get("floor") or observation["player"][PlayerFeature.FLOOR]),
    )
    terminal = False
    status = str(reset_info.get("episode_status", "running"))
    executed = 0
    error = ""
    for action in iter_trace_actions(trace):
        if terminal:
            error = "episode terminated before the recorded action sequence ended"
            break
        mask = observation.get("action_mask")
        if mask is None or np.asarray(mask).shape != (ACTION_COUNT,):
            error = "live replay observation has an invalid action mask"
            break
        if not bool(mask[action]):
            error = f"recorded action {action} is masked at replay turn {executed}"
            break
        observation, _, terminated, truncated, info = environment.step(action)
        turn_digests.append(normalized_observation_digest(observation))
        executed += 1
        terminal = bool(terminated or truncated)
        status = str(info.get("episode_status", "running"))
        actual_events.update(_event_counts(info.get("raw_events")))
        furthest = deeper_level(
            furthest,
            (
                int(info.get("zone") or observation["player"][PlayerFeature.ZONE]),
                int(info.get("floor") or observation["player"][PlayerFeature.FLOOR]),
            ),
        )
    expected_events = {str(key): int(value) for key, value in trace["event_counts"].items()}
    checks = {
        "all_actions_executed": executed == int(trace["turns"]),
        "terminal": terminal,
        "status": status == str(trace["status"]),
        "furthest_zone": furthest[0] == int(trace["furthest_zone"]),
        "furthest_floor": furthest[1] == int(trace["furthest_floor"]),
        "event_counts": dict(actual_events) == expected_events,
    }
    return {
        "trace_id": str(trace["trace_id"]),
        "seed": seed,
        "valid": not error and all(checks.values()),
        "error": error,
        "checks": checks,
        "expected": {
            "status": trace["status"],
            "turns": trace["turns"],
            "furthest_zone": trace["furthest_zone"],
            "furthest_floor": trace["furthest_floor"],
            "event_counts": expected_events,
        },
        "actual": {
            "status": status,
            "turns": executed,
            "furthest_zone": furthest[0],
            "furthest_floor": furthest[1],
            "event_counts": dict(actual_events),
        },
        "elapsed_seconds": time.monotonic() - started,
        "turn_digests": turn_digests,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def qualify_bank(arguments: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(arguments.bank.read_text(encoding="utf-8"))
    validate_demonstration_bank(bank)
    validate_demonstration_sources(bank)
    traces = list(bank["traces"])
    if not traces:
        raise ValueError("demonstration bank contains no successful traces")
    capacity = int(arguments.num_instances)
    config = SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=capacity,
        startup_timeout=arguments.startup_timeout,
        turn_timeout=arguments.turn_timeout,
        reset_timeout=arguments.reset_timeout,
        max_turns=max(int(trace["turns"]) for trace in traces) + 1,
        curriculum_commands_enabled=True,
        diagnostic_root=arguments.output.parent / "controller-diagnostics",
        affinity_policy=arguments.affinity,
    )
    results: list[dict[str, Any]] = []
    infrastructure_error = ""
    restarts = 0
    with AutoDancerSupervisor(config) as supervisor:
        assignments = [traces[index::capacity] for index in range(capacity)]

        def worker(slot: int) -> list[dict[str, Any]]:
            environment = supervisor.environment(supervisor.worker_ids[slot])
            return [replay_trace(environment, trace) for trace in assignments[slot]]

        try:
            with ThreadPoolExecutor(max_workers=capacity) as executor:
                for completed in executor.map(worker, range(capacity)):
                    results.extend(completed)
        except BaseException as error:
            infrastructure_error = f"{type(error).__name__}: {error}"
        restarts = sum(handle.restart_count for handle in supervisor.workers.values())
    results.sort(key=lambda result: (result["seed"], result["trace_id"]))
    valid = bool(
        not infrastructure_error
        and restarts == 0
        and len(results) == len(traces)
        and all(result["valid"] for result in results)
    )
    report = {
        "schema_version": 1,
        "kind": "qualified-live-action-traces-report-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "bank": str(arguments.bank.resolve()),
        "bank_sha256": bank["bank_sha256"],
        "num_instances": capacity,
        "trace_count": len(traces),
        "qualified_trace_count": sum(bool(result["valid"]) for result in results),
        "worker_restarts": restarts,
        "infrastructure_error": infrastructure_error,
        "valid": valid,
        "results": results,
    }
    _write_json_atomic(arguments.output, report)
    return report


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or qualify successful live action traces")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a hash-bound bank from episode journals")
    build.add_argument("--episodes", type=Path, action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    qualify = commands.add_parser("qualify", help="replay every bank trace in fresh live runs")
    qualify.add_argument("--game-dir", type=Path, required=True)
    qualify.add_argument("--mod-dir", type=Path, default=default_mod_dir())
    qualify.add_argument("--bank", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--num-instances", type=_positive_integer, default=4)
    qualify.add_argument("--startup-timeout", type=float, default=45.0)
    qualify.add_argument("--turn-timeout", type=float, default=10.0)
    qualify.add_argument("--reset-timeout", type=float, default=30.0)
    qualify.add_argument("--affinity", choices=("auto", "none", "spread"), default="auto")
    arguments = parser.parse_args()
    if arguments.command == "build":
        payload = build_demonstration_bank(arguments.episodes)
        write_demonstration_bank(arguments.output, payload)
        summary = {
            "output": str(arguments.output),
            "bank_sha256": payload["bank_sha256"],
            "trace_count": len(payload["traces"]),
        }
    else:
        if arguments.mod_dir is None:
            parser.error("--mod-dir is required when LOCALAPPDATA is unavailable")
        report = qualify_bank(arguments)
        summary = {
            "output": str(arguments.output),
            "valid": report["valid"],
            "trace_count": report["trace_count"],
            "qualified_trace_count": report["qualified_trace_count"],
        }
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
