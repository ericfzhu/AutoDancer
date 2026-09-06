"""Bounded live frozen-policy scaling; does not optimize or promote a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.training.async_collector import VersionedAsyncRolloutCollector
from autodancer.training.model import model_from_spec


class ResourceSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        psutil.cpu_percent()
        while not self.stop.is_set():
            sample = {
                "cpu_percent": psutil.cpu_percent(),
                "available_memory_bytes": float(psutil.virtual_memory().available),
            }
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    check=True,
                )
                utilization, memory = result.stdout.splitlines()[0].split(",")
                sample.update(gpu_percent=float(utilization), gpu_memory_mib=float(memory))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.samples.append(sample)
            self.stop.wait(1)

    def __enter__(self) -> ResourceSampler:
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join()

    def report(self) -> dict[str, Any]:
        return {
            "sample_count": len(self.samples),
            "mean_cpu_percent": statistics.mean(s["cpu_percent"] for s in self.samples)
            if self.samples
            else None,
            "mean_gpu_percent": statistics.mean(
                s["gpu_percent"] for s in self.samples if "gpu_percent" in s
            )
            if any("gpu_percent" in s for s in self.samples)
            else None,
            "minimum_available_memory_bytes": min(s["available_memory_bytes"] for s in self.samples)
            if self.samples
            else None,
            "samples": self.samples,
        }


def memory_requirement(capacity: int, worker_bytes: int, reserve_bytes: int) -> int:
    return capacity * worker_bytes + reserve_bytes


def measure(args: argparse.Namespace, capacity: int, mode: str) -> dict[str, Any]:
    trial = args.output / f"{mode}-{capacity}"
    trial.mkdir()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_spec(payload["architecture"], initialize=False).to(device).eval()
    model.load_state_dict(payload["model"])
    del payload
    config = SupervisorConfig(
        game_dir=args.game_dir,
        mod_dir=args.mod_dir,
        num_instances=capacity,
        affinity_policy="none",
        steam_presence_worker=0,
        diagnostic_root=trial / "controller-diagnostics",
    )
    launch_started = time.monotonic()
    with AutoDancerSupervisor(config) as supervisor:
        startup = time.monotonic() - launch_started
        environment = AutoDancerVectorEnv(supervisor)
        collector = None
        try:
            reset_started = time.monotonic()
            collector = VersionedAsyncRolloutCollector(
                environment,
                model,
                device=device,
                seed=args.seed,
                batch_delay=args.batch_delay_ms / 1000,
                inference_transfer_mode=mode,
                action_contract="map-navigation-prior-v1",
            )
            initial_reset = time.monotonic() - reset_started
            collector.collect(args.warmup_steps)
            before = supervisor.health()
            python_process = psutil.Process()
            before_python_cpu = sum(python_process.cpu_times()[:2])
            rollouts = []
            with ResourceSampler() as sampler:
                started = time.monotonic()
                for _ in range(args.fragments):
                    rollout = collector.collect(args.steps)
                    del rollout
                    rollouts.append(dict(collector.last_runtime_metrics))
                    collector.completed_episodes.clear()
                    if collector.last_runtime_metrics["collector_recoveries_total"]:
                        raise RuntimeError("Controller recovery invalidated scaling measurement")
                elapsed = time.monotonic() - started
            after = supervisor.health()
            scheduler = [r["inference_scheduler"] for r in rollouts]
            requests = sum(r["requests"] for r in scheduler)
            batches = sum(r["batches"] for r in scheduler)
            worker_cpu = sum(
                after[k].get("cpu_seconds", 0) - before[k].get("cpu_seconds", 0) for k in after
            )
            result = {
                "capacity": capacity,
                "transfer_mode": mode,
                "valid": True,
                "startup_seconds": startup,
                "initial_reset_seconds": initial_reset,
                "elapsed_seconds": elapsed,
                "learner_transitions": capacity * args.steps * args.fragments,
                "transitions_per_second": capacity * args.steps * args.fragments / elapsed,
                "mean_batch_size": requests / batches,
                "mean_inference_queue_ms": 1000
                * sum(r["queue_seconds"] for r in scheduler)
                / requests,
                "result_transfer_calls": sum(r["result_transfer_calls"] for r in scheduler),
                "worker_cpu_seconds": worker_cpu,
                "python_cpu_seconds": sum(python_process.cpu_times()[:2]) - before_python_cpu,
                "worker_rss_bytes": sum(h.get("working_set_bytes", 0) for h in after.values()),
                "max_worker_rss_bytes": max(h.get("working_set_bytes", 0) for h in after.values()),
                "worker_restarts": sum(h["restart_count"] for h in after.values()),
                "resources": sampler.report(),
                "rollouts": rollouts,
            }
            (trial / "result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            return result
        finally:
            if collector is not None:
                collector.close()
            environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, default=Path("mods/AutoDancer"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacities", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--fragments", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=103001)
    parser.add_argument("--batch-delay-ms", type=float, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reserve-gib", type=float, default=1.0)
    args = parser.parse_args()
    if min(*args.capacities, args.steps, args.fragments, args.warmup_steps) <= 0:
        parser.error("capacities, steps, fragments, and warmup must be positive")
    if args.reserve_gib < 0 or args.batch_delay_ms < 0:
        parser.error("reserve and batch delay cannot be negative")
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    args.output.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "scope": "Frozen real policy, normal Bard starts; no PPO updates or guide prefixes",
        "torch": torch.__version__,
        "cpu_count": psutil.cpu_count(),
        "memory_total_bytes": psutil.virtual_memory().total,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": [],
    }
    worker_bytes = int(1.25 * 2**30)
    reserve = int(args.reserve_gib * 2**30)
    for capacity in sorted(set(args.capacities)):
        for mode in ("legacy", "packed"):
            available = psutil.virtual_memory().available
            required = memory_requirement(capacity, worker_bytes, reserve)
            if available < required:
                result = {
                    "capacity": capacity,
                    "transfer_mode": mode,
                    "valid": False,
                    "skipped": "insufficient available RAM for estimated workers plus reserve",
                    "available_bytes": available,
                    "required_bytes": required,
                }
            else:
                try:
                    result = measure(args, capacity, mode)
                    worker_bytes = max(worker_bytes, int(result["max_worker_rss_bytes"] * 1.1))
                except Exception as error:
                    result = {
                        "capacity": capacity,
                        "transfer_mode": mode,
                        "valid": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
            report["results"].append(result)
            valid = [r for r in report["results"] if r["valid"]]
            report["best_measured"] = (
                {
                    key: max(valid, key=lambda r: r["transitions_per_second"])[key]
                    for key in ("capacity", "transfer_mode", "transitions_per_second")
                }
                if valid
                else None
            )
            (args.output / "scaling.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(
                json.dumps({k: v for k, v in result.items() if k not in {"resources", "rollouts"}}),
                flush=True,
            )
            if "error" in result:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
