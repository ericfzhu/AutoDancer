"""Completion metrics for simulator and live policy evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autodancer.envs.sim import AutoDancerSimEnv
from autodancer.training.curriculum import SEED_RANGES

Policy = Callable[[dict[str, np.ndarray]], int]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    completed: bool
    furthest_zone: int
    furthest_floor: int
    deaths: int
    turns: int
    game_score: int


def run_episode(
    environment: Any,
    policy: Policy,
    seed: int,
    *,
    max_steps: int | None = None,
) -> EpisodeResult:
    reset_policy = getattr(policy, "reset", None)
    if callable(reset_policy):
        reset_policy()
    observation, info = environment.reset(seed=seed)
    initial_status = str(info.get("episode_status", "running"))
    terminated = initial_status in {"won", "dead"}
    truncated = initial_status == "aborted"
    steps = 0
    while not (terminated or truncated):
        if max_steps is not None and steps >= max_steps:
            info = dict(info)
            info["episode_status"] = "aborted"
            info["turns"] = steps
            break
        action = int(policy(observation))
        observation, _, terminated, truncated, info = environment.step(action)
        steps += 1

    unwrapped = getattr(environment, "unwrapped", environment)
    state = getattr(unwrapped, "state", None)
    completed = (
        info.get("episode_status") == "won"
        or bool(state is not None and state.won)
        or any(
            event.get("kind") == "success"
            and event.get("data", {}).get("task_complete", False)
            for event in info.get("raw_events", [])
        )
    )
    return EpisodeResult(
        seed=int((info.get("seed") if seed is None else seed) or 0),
        completed=completed,
        furthest_zone=int(info.get("zone", 0) or 0),
        furthest_floor=int(info.get("floor", 0) or 0),
        deaths=int(info.get("deaths", 0) or 0),
        turns=int(info.get("turns", steps) or steps),
        game_score=int(info.get("game_score", 0) or 0),
    )


def summarize(results: Iterable[EpisodeResult], source: str) -> dict[str, Any]:
    episodes = list(results)
    if not episodes:
        raise ValueError("At least one episode is required")
    return {
        "source": source,
        "episodes": len(episodes),
        "completion_rate": sum(result.completed for result in episodes) / len(episodes),
        "furthest_zone": max(result.furthest_zone for result in episodes),
        "furthest_floor": max(
            (result.furthest_zone - 1) * 4 + result.furthest_floor for result in episodes
        ),
        "deaths": sum(result.deaths for result in episodes),
        "mean_turns": sum(result.turns for result in episodes) / len(episodes),
        "mean_game_score": sum(result.game_score for result in episodes) / len(episodes),
        "results": [asdict(result) for result in episodes],
    }


class MaskedRandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation: dict[str, np.ndarray]) -> int:
        legal = np.flatnonzero(observation["action_mask"])
        if not len(legal):
            raise RuntimeError("The observation masks every action")
        return int(self.rng.choice(legal))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an unseen-seed random baseline")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    seeds = list(SEED_RANGES["test"][: arguments.episodes])
    policy = MaskedRandomPolicy(seed=2026)
    results = [
        run_episode(AutoDancerSimEnv(task="all_zones"), policy, seed) for seed in seeds
    ]
    report = summarize(results, source="simulator")
    payload = json.dumps(report, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
