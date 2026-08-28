"""Reset-only live boss identity calibration.

This module deliberately has no policy or environment-step path.  It is used to
construct boss-stratified seed banks without observing gameplay outcomes.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from autodancer.constants import BossType, PlayerFeature
from autodancer.curriculum import EpisodeResetSpec
from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.experiments.tracking import validate_qualification_freshness
from autodancer.live.bridge import CURRICULUM_PROFILES
from autodancer.live.protocol import SCHEMA_VERSION, SUPPORTED_GAME_VERSION, SUPPORTED_STEAM_BUILD
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig


class ResetOnlyVectorEnvironment(Protocol):
    num_envs: int
    worker_ids: list[str]
    infrastructure_events: list[dict[str, Any]]

    def reset(
        self,
        seeds: list[int],
        options: list[dict[str, Any] | None] | None = None,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]: ...


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 or seed >= 2**31 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be integers in [0, 2^31)")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("calibration seeds must be unique")
    return seeds


def _padding_seeds(count: int, excluded: set[int]) -> list[int]:
    result: list[int] = []
    candidate = 2**31 - 1
    while len(result) < count:
        if candidate not in excluded:
            result.append(candidate)
        candidate -= 1
    return result


def collect_boss_identities(
    environment: ResetOnlyVectorEnvironment,
    seeds: list[int],
    reset_spec: EpisodeResetSpec,
) -> list[dict[str, Any]]:
    """Observe boss identity immediately after reset, without taking an action."""
    if environment.num_envs <= 0:
        raise ValueError("reset-only calibration requires at least one worker")
    if len(environment.worker_ids) != environment.num_envs:
        raise ValueError("worker identity count does not match vector capacity")

    results: list[dict[str, Any]] = []
    seed_set = set(seeds)
    for offset in range(0, len(seeds), environment.num_envs):
        real_batch = seeds[offset : offset + environment.num_envs]
        padding = _padding_seeds(environment.num_envs - len(real_batch), seed_set)
        batch = [*real_batch, *padding]
        options = [reset_spec.reset_options() for _ in batch]
        observations, infos = environment.reset(batch, options=options)
        if len(infos) != environment.num_envs:
            raise ValueError("reset returned the wrong number of worker records")
        players = observations.get("player")
        if players is None or len(players) != environment.num_envs:
            raise ValueError("reset returned malformed player observations")

        for slot, seed in enumerate(real_batch):
            info = infos[slot]
            worker_id = environment.worker_ids[slot]
            observed_seed = int(info.get("seed", -1))
            if observed_seed != seed:
                raise ValueError(f"seed mismatch: requested {seed}, observed {observed_seed}")
            if info.get("instance_id") != worker_id:
                raise ValueError(
                    f"worker mismatch: expected {worker_id}, observed {info.get('instance_id')}"
                )
            if info.get("character") != "Bard":
                raise ValueError(f"boss identity reset did not start Bard for seed {seed}")
            if info.get("episode_status") != "running":
                raise ValueError(f"boss identity reset was terminal for seed {seed}")
            if info.get("curriculum_reset") != reset_spec.as_dict():
                raise ValueError(f"curriculum identity mismatch for seed {seed}")

            observation_type = int(players[slot, PlayerFeature.TASK])
            info_type = int(info.get("boss_type", -1))
            if observation_type != info_type:
                raise ValueError(f"boss identity disagreement for seed {seed}")
            try:
                boss_type = BossType(observation_type)
            except ValueError as error:
                raise ValueError(
                    f"unknown boss identity {observation_type} for seed {seed}"
                ) from error
            if boss_type is BossType.NONE:
                raise ValueError(f"no boss identity was present after reset for seed {seed}")

            results.append(
                {
                    "seed": seed,
                    "boss_type": int(boss_type),
                    "boss_name": boss_type.name,
                    "instance_id": worker_id,
                    "run_id": str(info.get("run_id", "")),
                    "session_id": str(info.get("session_id", "")),
                    "launch_id": str(info.get("launch_id", "")),
                    "curriculum_reset": reset_spec.as_dict(),
                }
            )

    if [result["seed"] for result in results] != seeds:
        raise ValueError("boss identity results did not preserve requested seed order")
    return results


def _load_qualification(path: Path, game_dir: Path, mod_dir: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"controller qualification is unavailable or malformed: {path}") from error
    if report.get("passed") is not True:
        raise ValueError("reset-only calibration requires a passed controller qualification")
    validate_qualification_freshness(report, game_dir=game_dir, mod_dir=mod_dir)
    return report


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    game_dir = arguments.game_dir.resolve()
    mod_dir = arguments.mod_dir.resolve()
    _load_qualification(arguments.controller_qualification, game_dir, mod_dir)
    reset_spec = EpisodeResetSpec(
        "boss-identity",
        arguments.curriculum_start_level,
        arguments.curriculum_target_level,
        arguments.curriculum_profile,
    )
    config = SupervisorConfig(
        game_dir=game_dir,
        mod_dir=mod_dir,
        num_instances=arguments.num_instances,
        max_turns=2,
        startup_timeout=arguments.startup_timeout,
        reset_timeout=arguments.reset_timeout,
        affinity_policy=arguments.affinity,
        diagnostic_root=arguments.output.parent / "controller-diagnostics",
        curriculum_start_level=reset_spec.start_level,
        curriculum_target_level=reset_spec.target_level,
        curriculum_profile=reset_spec.profile,
    )
    with AutoDancerSupervisor(config) as supervisor:
        environment = AutoDancerVectorEnv(supervisor)
        try:
            results = collect_boss_identities(environment, arguments.seeds, reset_spec)
            infrastructure_events = list(environment.infrastructure_events)
            worker_restarts = sum(worker.restart_count for worker in supervisor.workers.values())
        finally:
            environment.close()

    report = {
        "schema_version": 1,
        "kind": "boss-identity-calibration-v1",
        "disclosure": "reset boss identity only; no gameplay action was selected or issued",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_schema_version": SCHEMA_VERSION,
        "game_version": SUPPORTED_GAME_VERSION,
        "steam_build": SUPPORTED_STEAM_BUILD,
        "character": "Bard",
        "mode": "AllZonesSeededCurriculum",
        "num_instances": arguments.num_instances,
        "seeds": arguments.seeds,
        "curriculum_reset": reset_spec.as_dict(),
        "controller_qualification": str(arguments.controller_qualification.resolve()),
        "controller_qualification_sha256": sha256_file(arguments.controller_qualification),
        "controller_valid": not infrastructure_events and worker_restarts == 0,
        "worker_restarts": worker_restarts,
        "infrastructure_events": infrastructure_events,
        "results": results,
    }
    atomic_json(arguments.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Observe live boss identities after reset without taking policy actions"
    )
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-instances", type=int, default=8)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--curriculum-start-level", type=int, required=True)
    parser.add_argument("--curriculum-target-level", type=int)
    parser.add_argument("--curriculum-profile", choices=CURRICULUM_PROFILES, default="normal")
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--reset-timeout", type=float, default=30.0)
    parser.add_argument("--affinity", choices=("auto", "none", "spread"), default="auto")
    parser.add_argument("--controller-qualification", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.num_instances <= 0:
        parser.error("--num-instances must be positive")
    report = run(arguments)
    print(
        json.dumps(
            {
                "controller_valid": report["controller_valid"],
                "output": str(arguments.output),
                "resets": len(report["results"]),
            },
            sort_keys=True,
        )
    )
    return 0 if report["controller_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
