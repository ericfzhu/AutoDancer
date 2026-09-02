"""Deterministic, checkpointable training-level seed schedules."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrainingSeedSchedule:
    """Own one independent reset-seed stream per live worker slot."""

    base_seed: int
    slots: int
    pool: tuple[int, ...] = ()
    _rngs: list[np.random.Generator] = field(init=False, repr=False)
    _draws: list[int] = field(init=False, repr=False)
    _pool_counts: dict[int, int] = field(init=False, repr=False)
    _unattributed_draws: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.slots <= 0:
            raise ValueError("slots must be positive")
        if len(set(self.pool)) != len(self.pool):
            raise ValueError("training seed pool must not contain duplicates")
        if any(seed < 0 or seed >= 2**31 for seed in self.pool):
            raise ValueError("training seeds must be integers in [0, 2^31)")
        sequence = np.random.SeedSequence(int(self.base_seed))
        self._rngs = [
            np.random.default_rng(child) for child in sequence.spawn(self.slots)
        ]
        self._draws = [0] * self.slots
        self._pool_counts = {int(seed): 0 for seed in self.pool}
        self._unattributed_draws = 0

    @property
    def mode(self) -> str:
        return "uniform-pool-v1" if self.pool else "unbounded-random-v1"

    def next(self, slot: int) -> int:
        if not 0 <= slot < self.slots:
            raise IndexError(f"Seed-schedule slot {slot} is outside capacity {self.slots}")
        rng = self._rngs[slot]
        if self.pool:
            seed = self.pool[int(rng.integers(0, len(self.pool)))]
            self._pool_counts[int(seed)] += 1
        else:
            seed = int(rng.integers(0, 2**31))
        self._draws[slot] += 1
        return int(seed)

    def draw_count(self, slot: int) -> int:
        """Return the checkpointed number of resets issued for one actor slot."""

        if not 0 <= slot < self.slots:
            raise IndexError(f"Seed-schedule slot {slot} is outside capacity {self.slots}")
        return int(self._draws[slot])

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "base_seed": int(self.base_seed),
            "slots": int(self.slots),
            "pool": list(self.pool),
            "draws": list(self._draws),
            "pool_counts": {
                str(seed): int(self._pool_counts[seed]) for seed in sorted(self._pool_counts)
            },
            "unattributed_draws": int(self._unattributed_draws),
            "rng_states": [deepcopy(rng.bit_generator.state) for rng in self._rngs],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema_version": 1,
            "mode": self.mode,
            "base_seed": int(self.base_seed),
            "slots": int(self.slots),
            "pool": list(self.pool),
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Training seed schedule state does not match this run")
        draws = [int(value) for value in state.get("draws", [])]
        rng_states = list(state.get("rng_states", []))
        if len(draws) != self.slots or len(rng_states) != self.slots:
            raise ValueError("Training seed schedule state has the wrong capacity")
        if any(value < 0 for value in draws):
            raise ValueError("Training seed schedule draw counts must be non-negative")
        for rng, rng_state in zip(self._rngs, rng_states, strict=True):
            rng.bit_generator.state = deepcopy(rng_state)
        self._draws = draws
        raw_counts = state.get("pool_counts")
        if raw_counts is None:
            # Old checkpoints remain behaviorally resumable, but their historical
            # per-seed draws cannot be reconstructed from RNG state alone.
            self._pool_counts = {int(seed): 0 for seed in self.pool}
            self._unattributed_draws = sum(draws) if self.pool else 0
            return
        if not isinstance(raw_counts, dict):
            raise ValueError("Training seed pool counts must be an object")
        try:
            counts = {int(seed): int(count) for seed, count in raw_counts.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Training seed pool counts must contain integers") from error
        if set(counts) != set(self.pool) or any(count < 0 for count in counts.values()):
            raise ValueError("Training seed pool counts do not match this run")
        unattributed = int(state.get("unattributed_draws", 0))
        if not self.pool:
            if counts or unattributed != 0:
                raise ValueError("Unbounded training schedules cannot contain pool counts")
            self._pool_counts = {}
            self._unattributed_draws = 0
            return
        if unattributed < 0 or sum(counts.values()) + unattributed != sum(draws):
            raise ValueError("Training seed pool counts do not match total draws")
        self._pool_counts = counts
        self._unattributed_draws = unattributed


@dataclass(slots=True)
class ResetConditionedTrainingSeedSchedule:
    """Own an independent finite seed stream for every curriculum reset kind."""

    base_seed: int
    slots: int
    pools: dict[str, tuple[int, ...]]
    _schedules: dict[str, TrainingSeedSchedule] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.slots <= 0:
            raise ValueError("slots must be positive")
        if not self.pools:
            raise ValueError("reset-conditioned seed pools must not be empty")
        normalized: dict[str, tuple[int, ...]] = {}
        for reset_id, pool in self.pools.items():
            identifier = str(reset_id)
            if not identifier or any(
                not (character.isalnum() or character in "-_") for character in identifier
            ):
                raise ValueError("reset-conditioned seed pool ids are invalid")
            values = tuple(int(seed) for seed in pool)
            if not values:
                raise ValueError(f"reset-conditioned seed pool {identifier!r} is empty")
            if len(set(values)) != len(values):
                raise ValueError(
                    f"reset-conditioned seed pool {identifier!r} contains duplicates"
                )
            if any(seed < 0 or seed >= 2**31 for seed in values):
                raise ValueError("training seeds must be integers in [0, 2^31)")
            normalized[identifier] = values
        self.pools = {key: normalized[key] for key in sorted(normalized)}
        self._schedules = {}
        for reset_id, pool in self.pools.items():
            digest = hashlib.sha256(reset_id.encode("utf-8")).digest()
            salt = int.from_bytes(digest[:8], "little")
            child_seed = int(self.base_seed) ^ salt
            self._schedules[reset_id] = TrainingSeedSchedule(child_seed, self.slots, pool)

    @property
    def mode(self) -> str:
        return "reset-conditioned-uniform-pools-v1"

    def next(self, slot: int, reset_id: str) -> int:
        try:
            schedule = self._schedules[str(reset_id)]
        except KeyError as error:
            raise ValueError(
                f"curriculum reset {reset_id!r} has no training seed pool"
            ) from error
        return schedule.next(slot)

    def draw_count(self, slot: int) -> int:
        if not 0 <= slot < self.slots:
            raise IndexError(f"Seed-schedule slot {slot} is outside capacity {self.slots}")
        return sum(schedule.draw_count(slot) for schedule in self._schedules.values())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "base_seed": int(self.base_seed),
            "slots": int(self.slots),
            "pools": {key: list(value) for key, value in self.pools.items()},
            "draws": [self.draw_count(slot) for slot in range(self.slots)],
            "schedules": {
                key: schedule.state_dict() for key, schedule in self._schedules.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema_version": 1,
            "mode": self.mode,
            "base_seed": int(self.base_seed),
            "slots": int(self.slots),
            "pools": {key: list(value) for key, value in self.pools.items()},
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Reset-conditioned seed schedule state does not match this run")
        schedules = state.get("schedules")
        if not isinstance(schedules, dict) or set(schedules) != set(self._schedules):
            raise ValueError("Reset-conditioned seed schedule state has the wrong pools")
        for reset_id, schedule in self._schedules.items():
            child_state = schedules[reset_id]
            if not isinstance(child_state, dict):
                raise ValueError("Reset-conditioned child schedule state must be an object")
            schedule.load_state_dict(child_state)
        expected_draws = [self.draw_count(slot) for slot in range(self.slots)]
        if [int(value) for value in state.get("draws", [])] != expected_draws:
            raise ValueError("Reset-conditioned seed schedule draw counts do not match")


def parse_training_seed_pool(value: str) -> tuple[int, ...]:
    """Parse either `start-end` or a comma-separated finite seed pool."""
    text = value.strip()
    try:
        if "," not in text and text.count("-") == 1:
            start_text, end_text = text.split("-", 1)
            start, end = int(start_text), int(end_text)
            seeds = tuple(range(start, end + 1))
        else:
            seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("training seed pool must be `start-end` or comma-separated") from error
    if not seeds:
        raise ValueError("training seed pool must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seed pool must not contain duplicates")
    if any(seed < 0 or seed >= 2**31 for seed in seeds):
        raise ValueError("training seeds must be integers in [0, 2^31)")
    return seeds


def load_reset_conditioned_seed_pools(path: str | Path) -> dict[str, tuple[int, ...]]:
    """Load exact finite seed pools keyed by curriculum reset id."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not load reset-conditioned seed pools {source}: {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("reset-conditioned seed pools must use schema_version 1")
    if set(payload) != {"schema_version", "pools"}:
        raise ValueError("reset-conditioned seed pools contain unknown fields")
    raw_pools = payload.get("pools")
    if not isinstance(raw_pools, dict) or not raw_pools:
        raise ValueError("reset-conditioned seed pools must contain a non-empty pools object")
    pools: dict[str, tuple[int, ...]] = {}
    for reset_id, raw_pool in raw_pools.items():
        if not isinstance(raw_pool, list):
            raise ValueError(f"reset-conditioned seed pool {reset_id!r} must be a list")
        try:
            pool = tuple(int(seed) for seed in raw_pool)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"reset-conditioned seed pool {reset_id!r} must contain integers"
            ) from error
        pools[str(reset_id)] = pool
    # Construct once for centralized identifier, range, duplicate, and emptiness validation.
    return ResetConditionedTrainingSeedSchedule(0, 1, pools).pools
