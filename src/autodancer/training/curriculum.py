"""Smooth task mixture and non-overlapping deterministic seed sets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from autodancer.envs.sim import AutoDancerSimEnv
from autodancer.tasks import TASKS

TASK_ORDER = tuple(TASKS)
SEED_RANGES: dict[str, range] = {
    "train": range(0, 1_000_000),
    "validation": range(1_000_000, 1_010_000),
    "test": range(2_000_000, 2_010_000),
}


def fixed_seed(split: str, stream: str, episode: int) -> int:
    choices = SEED_RANGES[split]
    digest = hashlib.blake2b(
        f"autodancer:{split}:{stream}:{episode}".encode(), digest_size=8
    ).digest()
    return choices.start + int.from_bytes(digest, "little") % len(choices)


@dataclass(slots=True)
class AdaptiveTaskMixer:
    """Prefer tasks near the policy's learning frontier without a hard gate."""

    learning_rate: float = 0.05
    target_success: float = 0.70
    success_rates: dict[str, float] = field(
        default_factory=lambda: {
            task: initial
            for task, initial in zip(
                TASK_ORDER, (0.50, 0.30, 0.15, 0.08, 0.03, 0.01), strict=True
            )
        }
    )

    def update(self, task: str, success: bool) -> None:
        old = self.success_rates[task]
        self.success_rates[task] = old + self.learning_rate * (float(success) - old)

    def probabilities(self) -> np.ndarray:
        rates = np.asarray([self.success_rates[task] for task in TASK_ORDER])
        difficulty = np.arange(len(TASK_ORDER), dtype=np.float64)
        frontier = np.exp(-4.0 * np.abs(rates - self.target_success))
        practice = 0.08 + 0.15 * (1.0 - rates)
        difficulty_prior = np.exp(-0.22 * difficulty)
        weights = (frontier + practice) * difficulty_prior
        return weights / weights.sum()

    def choose(self, rng: np.random.Generator) -> str:
        index = int(rng.choice(len(TASK_ORDER), p=self.probabilities()))
        return TASK_ORDER[index]


class CurriculumEnv(gym.Env[dict[str, np.ndarray], int]):
    """Switch task at episode reset while keeping one policy and one schema."""

    metadata = AutoDancerSimEnv.metadata

    def __init__(
        self,
        stream: str = "worker-0-env-0",
        split: str = "train",
        render_mode: str | None = None,
    ) -> None:
        self.stream = stream
        self.split = split
        self.render_mode = render_mode
        self.episode = 0
        stream_seed = fixed_seed(split, stream, 0)
        self.rng = np.random.default_rng(stream_seed)
        self.mixer = AdaptiveTaskMixer()
        self.current_task = "navigation"
        self.environment = AutoDancerSimEnv(task=self.current_task, render_mode=render_mode)
        self.action_space = self.environment.action_space
        self.observation_space = self.environment.observation_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.current_task = self.mixer.choose(self.rng)
        self.environment = AutoDancerSimEnv(
            task=self.current_task, render_mode=self.render_mode
        )
        episode_seed = (
            seed
            if seed is not None
            else fixed_seed(self.split, self.stream, self.episode)
        )
        self.episode += 1
        observation, info = self.environment.reset(seed=episode_seed, options=options)
        info["curriculum_task"] = self.current_task
        info["curriculum_probabilities"] = self.mixer.probabilities().tolist()
        return observation, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.environment.step(action)
        if terminated or truncated:
            success = any(
                event.get("kind") == "success"
                and event.get("data", {}).get("task_complete", False)
                for event in info["raw_events"]
            )
            self.mixer.update(self.current_task, success)
            info["task_success"] = int(success)
            info["recent_task_success"] = self.mixer.success_rates[self.current_task]
        info["curriculum_task"] = self.current_task
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self.environment.render()

    def close(self) -> None:
        self.environment.close()
