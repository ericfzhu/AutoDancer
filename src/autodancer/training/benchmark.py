"""Measure fixed-capacity live-worker throughput without PPO updates."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.training.train import default_mod_dir


def _run_slot(environment: AutoDancerVectorEnv, index: int, steps: int) -> dict[str, Any]:
    worker_id = environment.worker_ids[index]
    worker = environment.environments[worker_id]
    rng = np.random.default_rng(10_000 + index)
    observation, _ = worker.reset(seed=int(rng.integers(0, 2**31)))
    latencies: list[float] = []
    completed = 0
    restarts = 0
    slot_started = time.monotonic()
    while completed < steps:
        valid = np.flatnonzero(observation["action_mask"])
        action = int(valid[completed % len(valid)])
        started = time.monotonic()
        try:
            observation, reward, terminated, truncated, info = worker.step(action)
            del reward
        except Exception:
            observation, _ = environment.recover(index, int(rng.integers(0, 2**31)))
            worker = environment.environments[worker_id]
            restarts += 1
            continue
        latencies.append(time.monotonic() - started)
        handle = environment.supervisor.workers[worker_id]
        handle.last_latency = latencies[-1]
        handle.episode_status = str(info["episode_status"])
        acknowledgement = info.get("bridge") or {}
        handle.last_acknowledged_command = int(
            acknowledgement.get("command_id", handle.last_acknowledged_command)
        )
        completed += 1
        if terminated or truncated:
            observation, _ = worker.reset(seed=int(rng.integers(0, 2**31)))
    elapsed = time.monotonic() - slot_started
    return {
        "worker_id": worker_id,
        "latencies": latencies,
        "restarts": restarts,
        "elapsed_seconds": elapsed,
        "transitions_per_second": completed / max(elapsed, 1e-9),
    }


def benchmark_capacity(arguments: argparse.Namespace, capacity: int) -> dict[str, Any]:
    config = SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=capacity,
        startup_timeout=arguments.startup_timeout,
        turn_timeout=arguments.turn_timeout,
        affinity_policy=arguments.affinity,
        diagnostic_root=arguments.run_dir / "controller-diagnostics",
    )
    with AutoDancerSupervisor(config) as supervisor:
        environment = AutoDancerVectorEnv(supervisor)
        started = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=capacity) as executor:
                results = list(
                    executor.map(
                        lambda index: _run_slot(environment, index, arguments.steps),
                        range(capacity),
                    )
                )
            elapsed = time.monotonic() - started
            latencies = np.asarray(
                [latency for result in results for latency in result["latencies"]],
                dtype=np.float64,
            )
            return {
                "num_instances": capacity,
                "steps": int(latencies.size),
                "elapsed_seconds": elapsed,
                "transitions_per_second": float(latencies.size / max(elapsed, 1e-9)),
                "per_worker_transitions_per_second": float(arguments.steps / max(elapsed, 1e-9)),
                "latency_p50_ms": float(np.percentile(latencies, 50) * 1000),
                "latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
                "latency_p99_ms": float(np.percentile(latencies, 99) * 1000),
                "worker_restarts": sum(result["restarts"] for result in results),
                "per_worker": {
                    result["worker_id"]: {
                        "transitions_per_second": result["transitions_per_second"],
                        "latency_p50_ms": float(np.percentile(result["latencies"], 50) * 1000),
                        "latency_p95_ms": float(np.percentile(result["latencies"], 95) * 1000),
                        "restarts": result["restarts"],
                    }
                    for result in results
                },
                "workers": supervisor.health(),
                "affinity": arguments.affinity,
            }
        finally:
            environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark live AutoDancer workers")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, default=default_mod_dir())
    parser.add_argument("--num-instances", type=int)
    parser.add_argument("--sweep", default="")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--turn-timeout", type=float, default=10.0)
    parser.add_argument("--affinity", choices=("auto", "none", "spread"), default="auto")
    arguments = parser.parse_args()
    if arguments.mod_dir is None:
        parser.error("--mod-dir is required when LOCALAPPDATA is unavailable")
    capacities = (
        [int(value) for value in arguments.sweep.split(",") if value.strip()]
        if arguments.sweep
        else [arguments.num_instances]
        if arguments.num_instances
        else []
    )
    if not capacities or any(value <= 0 for value in capacities) or arguments.steps <= 0:
        parser.error("provide positive --num-instances or --sweep values and --steps")
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for capacity in capacities:
        result = benchmark_capacity(arguments, capacity)
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    recommended = max(results, key=lambda result: result["transitions_per_second"])
    report = {
        "schema_version": 1,
        "results": results,
        "recommended_num_instances": recommended["num_instances"],
        "acceptance_baseline": 5.315918163931681,
        "acceptance_target": 10.631836327863362,
        "acceptance_passed": any(
            result["num_instances"] == 8 and result["transitions_per_second"] >= 10.631836327863362
            for result in results
        ),
    }
    destination = arguments.run_dir / "benchmark.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"benchmark_report": str(destination)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
