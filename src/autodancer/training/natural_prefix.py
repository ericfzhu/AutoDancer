"""Reachable-state handoff contracts for reverse-curriculum training.

A natural prefix is a sequence of ordinary, acknowledged game actions executed
by a guide policy.  The learner takes over the exact running process only after
the observed game state proves that a declared boundary was reached.  No game
state is synthesized and guide transitions are not PPO samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from autodancer.constants import ActorKind, BossType, GridChannel, PlayerFeature

NATURAL_PREFIX_RECURRENT_MODES = ("fresh", "warm")


class NaturalPrefixError(RuntimeError):
    """Raised when a legal guide prefix cannot reach its declared boundary."""


@dataclass(frozen=True, slots=True)
class NaturalPrefixConfig:
    """Immutable Death Metal handoff definition used by training and evaluation."""

    boss_type: int = int(BossType.DEATH_METAL)
    target_phase: int = 4
    max_guide_turns: int = 512
    max_attempts: int = 8
    deterministic_guide: bool = False
    guide_policy_seed: int = 0
    recurrent_state_mode: str = "fresh"

    def __post_init__(self) -> None:
        if self.boss_type != int(BossType.DEATH_METAL):
            raise ValueError("The first natural-prefix contract supports Death Metal only")
        if self.target_phase not in {2, 3, 4}:
            raise ValueError("target_phase must be 2, 3, or 4")
        if self.max_guide_turns <= 0 or self.max_attempts <= 0:
            raise ValueError("natural-prefix turn and attempt limits must be positive")
        if self.recurrent_state_mode not in NATURAL_PREFIX_RECURRENT_MODES:
            raise ValueError(
                "recurrent_state_mode must be one of " + ", ".join(NATURAL_PREFIX_RECURRENT_MODES)
            )

    @property
    def target_health(self) -> int:
        return {2: 6, 3: 4, 4: 2}[self.target_phase]

    def specification(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "death-metal-natural-prefix-v1",
            **asdict(self),
            "target_health": self.target_health,
            "state_semantics": "ordinary-engine-transitions-only",
            "guide_transitions_in_ppo": False,
        }


@dataclass(slots=True)
class DeathMetalPhaseTracker:
    """Require visible, damage-backed evidence of Death Metal phase conversion.

    Health alone is deliberately insufficient: the invalid historical profile
    directly set health without invoking Death Metal's conversion handler.  A
    valid boundary therefore requires the expected number of distinct observed
    boss entity types as well as cumulative boss damage and current health.
    Type-hash collisions can only make this conservative test fail closed.
    """

    config: NaturalPrefixConfig
    initial_health: int | None = None
    minimum_health: int | None = None
    boss_damage: int = 0
    observed_actor_types: set[int] = field(default_factory=set)
    observations_with_boss: int = 0

    @staticmethod
    def _visible_bosses(observation: Mapping[str, np.ndarray]) -> list[tuple[int, int, int]]:
        grid = observation["grid"]
        mask = (
            (grid[..., GridChannel.ACTOR_CLASS] == int(ActorKind.BOSS))
            & (grid[..., GridChannel.HEALTH] > 0)
            & (grid[..., GridChannel.VISIBILITY] == 2)
        )
        return [
            (
                int(grid[row, column, GridChannel.ACTOR_TYPE]),
                int(grid[row, column, GridChannel.HEALTH]),
                int(grid[row, column, GridChannel.MAX_HEALTH]),
            )
            for row, column in np.argwhere(mask)
        ]

    def observe(
        self,
        observation: Mapping[str, np.ndarray],
        info: Mapping[str, Any] | None = None,
    ) -> None:
        if int(observation["player"][PlayerFeature.TASK]) != self.config.boss_type:
            return
        bosses = self._visible_bosses(observation)
        if bosses:
            self.observations_with_boss += 1
            health = min(value[1] for value in bosses)
            self.initial_health = (
                health if self.initial_health is None else max(self.initial_health, health)
            )
            self.minimum_health = (
                health if self.minimum_health is None else min(self.minimum_health, health)
            )
            self.observed_actor_types.update(value[0] for value in bosses if value[0] != 0)
        for event in (info or {}).get("raw_events", []):
            data = event.get("data") or {}
            if event.get("kind") == "enemy_damage" and bool(data.get("boss")):
                self.boss_damage += max(int(event.get("amount", 0) or 0), 0)

    @property
    def reached(self) -> bool:
        health = self.minimum_health
        if self.initial_health is None or health is None:
            return False
        required_damage = max(self.initial_health - self.config.target_health, 1)
        return bool(
            health <= self.config.target_health
            and self.boss_damage >= required_damage
            and len(self.observed_actor_types) >= self.config.target_phase
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "boss_type": self.config.boss_type,
            "target_phase": self.config.target_phase,
            "target_health": self.config.target_health,
            "initial_health": self.initial_health,
            "minimum_health": self.minimum_health,
            "boss_damage": self.boss_damage,
            "observed_actor_types": sorted(self.observed_actor_types),
            "observations_with_boss": self.observations_with_boss,
            "reached": self.reached,
        }
