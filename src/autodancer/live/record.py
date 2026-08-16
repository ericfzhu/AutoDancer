"""Record versioned live conformance traces from the bounded symbolic explorer."""

from __future__ import annotations

import argparse
from pathlib import Path

from autodancer.envs.live import AutoDancerLiveEnv
from autodancer.live.explore import LiveExplorer
from autodancer.live.trace import TraceWriter
from autodancer.tasks import TASKS


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a bounded live conformance trace")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", choices=sorted(TASKS), default="all_zones")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--turn-timeout", type=float, default=5.0)
    parser.add_argument("--attach-existing", action="store_true")
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Expected-path glob to omit during comparison; repeat as needed",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--compare-observation",
        action="store_true",
        help="Treat the captured live observation as simulator-expected data",
    )
    parser.add_argument("--compare-reward", action="store_true")
    parser.add_argument("--compare-events", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_turns <= 0:
        parser.error("--max-turns must be positive")

    environment = AutoDancerLiveEnv(
        log_path=arguments.log_path,
        attach_existing=arguments.attach_existing,
        task=arguments.task,
        max_turns=arguments.max_turns,
        turn_timeout=arguments.turn_timeout,
    )
    try:
        observation, info = environment.reset()
        writer = TraceWriter(
            arguments.output,
            info=info,
            task=arguments.task,
            initial_observation=observation,
            ignored_paths=arguments.ignore,
            strict=arguments.strict,
            compare_observation=arguments.compare_observation,
            compare_reward=arguments.compare_reward,
            compare_events=arguments.compare_events,
            overwrite=arguments.overwrite,
        )
        if info.get("episode_status") != "running":
            print(arguments.output)
            return 0

        explorer = LiveExplorer()
        floor = info.get("floor")
        for turn in range(1, arguments.max_turns + 1):
            action = explorer.choose(observation)
            observation, reward, terminated, truncated, info = environment.step(action)
            writer.append(
                action=action,
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            next_floor = info.get("floor")
            if next_floor != floor:
                explorer.reset_level()
                floor = next_floor
            print(
                f"{turn}: {action.name} ({explorer.last_reason}) "
                f"floor={floor} pos={tuple(observation['player'][4:6])} "
                f"hp={int(observation['player'][0])} reward={reward:.3f} "
                f"status={info['episode_status']}"
            )
            if terminated or truncated:
                break
        print(arguments.output)
        return 0
    finally:
        environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
