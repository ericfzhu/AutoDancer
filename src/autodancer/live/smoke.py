"""Bounded native-worker protocol smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig


def default_mod_dir() -> Path | None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    return (
        Path(local_app_data) / "NecroDancer" / "mods" / "AutoDancer"
        if local_app_data
        else None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test native AutoDancer workers")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, default=default_mod_dir())
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    arguments = parser.parse_args()
    if arguments.mod_dir is None:
        parser.error("--mod-dir is required when LOCALAPPDATA is unavailable")
    config = SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=arguments.num_instances,
        startup_timeout=arguments.startup_timeout,
    )
    with AutoDancerSupervisor(config) as supervisor:
        environment = AutoDancerVectorEnv(supervisor)
        try:
            seeds = [arguments.seed + index for index in range(arguments.num_instances)]
            observation, infos = environment.reset(seeds)
            print(json.dumps({"phase": "reset", "infos": infos}, sort_keys=True))
            for step in range(arguments.steps):
                actions = []
                for mask in observation["action_mask"]:
                    legal = np.flatnonzero(mask)
                    actions.append(int(legal[step % len(legal)]))
                observation, rewards, terminated, truncated, infos = environment.step(actions)
                print(
                    json.dumps(
                        {
                            "phase": "step",
                            "step": step,
                            "actions": actions,
                            "rewards": rewards.tolist(),
                            "terminated": terminated.tolist(),
                            "truncated": truncated.tolist(),
                            "infos": infos,
                            "health": supervisor.health(),
                        },
                        sort_keys=True,
                    )
                )
                done = np.flatnonzero(terminated | truncated).tolist()
                if done:
                    resets = environment.reset_at(
                        done, [arguments.seed + 10000 + step * 100 + index for index in done]
                    )
                    for index, (replacement, _) in zip(done, resets, strict=True):
                        for key in observation:
                            observation[key][index] = replacement[key]
        finally:
            environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
