"""Deterministic, checkpointable per-episode curriculum reset schedules."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autodancer.live.bridge import CURRICULUM_PROFILES


@dataclass(frozen=True, slots=True)
class EpisodeResetSpec:
    """One fully identified live-game episode start specification."""

    id: str
    start_level: int = 1
    target_level: int | None = None
    profile: str = "normal"

    def __post_init__(self) -> None:
        valid_id = self.id and all(
            character.isalnum() or character in "-_" for character in self.id
        )
        if not valid_id:
            raise ValueError("curriculum reset id may contain only letters, digits, '-' and '_'")
        if self.start_level <= 0:
            raise ValueError("curriculum start_level must be positive")
        if self.profile not in CURRICULUM_PROFILES:
            raise ValueError("curriculum profile must be one of " + ", ".join(CURRICULUM_PROFILES))
        if self.profile != "normal" and self.start_level <= 1:
            raise ValueError("assisted curriculum profiles require a later start level")
        if self.target_level is not None and self.target_level <= self.start_level:
            raise ValueError("curriculum target_level must be after start_level")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_level": int(self.start_level),
            "target_level": self.target_level,
            "profile": self.profile,
        }

    def reset_options(self) -> dict[str, Any]:
        return {"curriculum": self.as_dict()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EpisodeResetSpec:
        unknown = set(value) - {"id", "start_level", "target_level", "profile", "weight"}
        if unknown:
            raise ValueError(f"unknown curriculum reset fields: {sorted(unknown)}")
        target = value.get("target_level")
        return cls(
            id=str(value.get("id", "default")),
            start_level=int(value.get("start_level", 1)),
            target_level=None if target is None else int(target),
            profile=str(value.get("profile", "normal")),
        )


@dataclass(frozen=True, slots=True)
class WeightedResetSpec:
    spec: EpisodeResetSpec
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("curriculum reset weights must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return {**self.spec.as_dict(), "weight": float(self.weight)}


def fixed_reset_spec(
    start_level: int,
    target_level: int | None,
    profile: str,
    *,
    id: str = "fixed",
) -> tuple[WeightedResetSpec, ...]:
    return (
        WeightedResetSpec(
            EpisodeResetSpec(id, int(start_level), target_level, str(profile)),
            1.0,
        ),
    )


def load_curriculum_mixture(path: str | Path) -> tuple[WeightedResetSpec, ...]:
    """Load an immutable weighted reset distribution from JSON."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load curriculum mixture {source}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("curriculum mixture must be a schema_version 1 JSON object")
    unknown = set(payload) - {"schema_version", "entries"}
    if unknown:
        raise ValueError(f"unknown curriculum mixture fields: {sorted(unknown)}")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("curriculum mixture entries must be a non-empty list")
    entries: list[WeightedResetSpec] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("each curriculum mixture entry must be an object")
        entries.append(
            WeightedResetSpec(
                EpisodeResetSpec.from_mapping(raw_entry),
                float(raw_entry.get("weight", 1.0)),
            )
        )
    ids = [entry.spec.id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("curriculum mixture entry ids must be unique")
    return tuple(entries)


@dataclass(slots=True)
class EpisodeCurriculumSchedule:
    """Own one timing-independent reset-distribution stream per worker slot."""

    base_seed: int
    slots: int
    entries: tuple[WeightedResetSpec, ...]
    _rngs: list[np.random.Generator] = field(init=False, repr=False)
    _draws: list[int] = field(init=False, repr=False)
    _selected: dict[str, int] = field(init=False, repr=False)
    _outcomes: dict[str, dict[str, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.slots <= 0:
            raise ValueError("curriculum schedule slots must be positive")
        if not self.entries:
            raise ValueError("curriculum schedule requires at least one entry")
        ids = [entry.spec.id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("curriculum schedule entry ids must be unique")
        sequence = np.random.SeedSequence([int(self.base_seed), 0xAD0C0DE])
        self._rngs = [np.random.default_rng(child) for child in sequence.spawn(self.slots)]
        self._draws = [0] * self.slots
        self._selected = {entry.spec.id: 0 for entry in self.entries}
        self._outcomes = {entry.spec.id: {} for entry in self.entries}

    @property
    def probabilities(self) -> np.ndarray:
        weights = np.asarray([entry.weight for entry in self.entries], dtype=np.float64)
        return weights / weights.sum()

    def next(self, slot: int) -> EpisodeResetSpec:
        if not 0 <= slot < self.slots:
            raise IndexError(f"Curriculum slot {slot} is outside capacity {self.slots}")
        index = int(self._rngs[slot].choice(len(self.entries), p=self.probabilities))
        spec = self.entries[index].spec
        self._draws[slot] += 1
        self._selected[spec.id] += 1
        return spec

    def record_outcome(self, spec: EpisodeResetSpec, status: str) -> None:
        if spec.id not in self._outcomes:
            raise ValueError(f"reset specification {spec.id!r} is not in this schedule")
        outcome = str(status)
        values = self._outcomes[spec.id]
        values[outcome] = values.get(outcome, 0) + 1

    def specification(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "weighted-per-episode-v1",
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            **self.specification(),
            "base_seed": int(self.base_seed),
            "slots": int(self.slots),
            "draws": list(self._draws),
            "selected": dict(self._selected),
            "outcomes": deepcopy(self._outcomes),
            "rng_states": [deepcopy(rng.bit_generator.state) for rng in self._rngs],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            **self.specification(),
            "base_seed": int(self.base_seed),
            "slots": int(self.slots),
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Curriculum schedule state does not match this run")
        draws = [int(value) for value in state.get("draws", [])]
        rng_states = list(state.get("rng_states", []))
        selected = {str(key): int(value) for key, value in dict(state.get("selected", {})).items()}
        outcomes = {
            str(key): {str(status): int(count) for status, count in dict(values).items()}
            for key, values in dict(state.get("outcomes", {})).items()
        }
        ids = {entry.spec.id for entry in self.entries}
        if len(draws) != self.slots or len(rng_states) != self.slots:
            raise ValueError("Curriculum schedule state has the wrong capacity")
        if set(selected) != ids or set(outcomes) != ids:
            raise ValueError("Curriculum schedule counters have the wrong entry ids")
        if any(value < 0 for value in draws) or any(value < 0 for value in selected.values()):
            raise ValueError("Curriculum schedule counts must be non-negative")
        if any(count < 0 for values in outcomes.values() for count in values.values()):
            raise ValueError("Curriculum outcome counts must be non-negative")
        for rng, rng_state in zip(self._rngs, rng_states, strict=True):
            rng.bit_generator.state = deepcopy(rng_state)
        self._draws = draws
        self._selected = selected
        self._outcomes = outcomes
