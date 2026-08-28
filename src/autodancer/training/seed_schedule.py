"""Deterministic, checkpointable training-level seed schedules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
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
